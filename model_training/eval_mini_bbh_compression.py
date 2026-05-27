#!/usr/bin/env python3

import argparse
import csv
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer


NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "n't",
}


@dataclass
class WordItem:
    index: int
    text: str
    start: int
    end: int
    p_keep: float = 0.0
    final_score: float = 0.0
    lemma: str = ""
    is_entity: bool = False
    is_number: bool = False
    is_date: bool = False
    is_negation: bool = False
    is_verb: bool = False
    is_aux: bool = False
    is_noun_chunk_root: bool = False
    is_stopword: bool = False
    redundancy: float = 0.0
    selected: bool = False


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_name", default="lukaemon/bbh")
    p.add_argument(
        "--tasks",
        default="boolean_expressions,date_understanding,logical_deduction_three_objects",
        help="Comma-separated BBH task configs, or 'auto'.",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--examples_per_task", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--compressor_model", required=True)

    p.add_argument(
        "--methods",
        default="none,every_second_word,random_keep,model_keep_ratio,stat_hybrid",
    )
    p.add_argument("--keep_ratio", type=float, default=0.6)
    p.add_argument("--max_input_tokens", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=32)

    p.add_argument("--spacy_model", default="en_core_web_sm")

    # stat hybrid weights
    p.add_argument("--verb_bonus", type=float, default=0.08)
    p.add_argument("--noun_chunk_root_bonus", type=float, default=0.08)
    p.add_argument("--entity_bonus", type=float, default=0.15)
    p.add_argument("--number_bonus", type=float, default=0.20)
    p.add_argument("--date_bonus", type=float, default=0.20)
    p.add_argument("--negation_bonus", type=float, default=0.30)
    p.add_argument("--redundancy_penalty", type=float, default=0.10)
    p.add_argument("--stopword_penalty", type=float, default=0.00)

    p.add_argument("--output_csv", default="results/eval/mini_bbh_compression_results.csv")
    p.add_argument("--output_jsonl", default="results/eval/mini_bbh_compression_results.jsonl")

    return p.parse_args()


def normalize_word(x: str) -> str:
    return re.sub(r"^\W+|\W+$", "", x.lower())


def normalize_answer(x: str) -> str:
    x = x.strip().lower()
    x = x.split("\n")[0].strip()

    # Remove common answer prefixes.
    x = re.sub(r"^(answer|final answer|the answer is)\s*[:\-]?\s*", "", x).strip()

    # Keep option letters if model outputs "(A)".
    x = x.strip()
    x = x.strip(string.whitespace)

    # Normalize punctuation lightly.
    x = x.replace("(", "").replace(")", "")
    x = x.strip(" .,:;!?\n\t\"'")

    return x


def is_correct(pred: str, target: str) -> bool:
    p = normalize_answer(pred)
    t = normalize_answer(target)

    if p == t:
        return True

    # For short targets like "yes", "no", "a", "b", "true", "false".
    pred_tokens = re.findall(r"[a-z0-9]+", p)
    target_tokens = re.findall(r"[a-z0-9]+", t)

    if len(target_tokens) == 1 and target_tokens[0] in pred_tokens[:5]:
        return True

    return False


def get_text_and_target(row: Dict) -> Tuple[str, str]:
    input_keys = ["input", "question", "prompt", "inputs"]
    target_keys = ["target", "answer", "label", "targets"]

    text = None
    target = None

    for k in input_keys:
        if k in row and row[k] is not None:
            text = str(row[k])
            break

    for k in target_keys:
        if k in row and row[k] is not None:
            target = row[k]
            if isinstance(target, list):
                target = target[0]
            target = str(target)
            break

    if text is None or target is None:
        raise ValueError(f"Cannot infer input/target keys from row keys: {list(row.keys())}")

    return text, target


def build_prompt(question: str) -> str:
    return (
        "Answer the following reasoning question. "
        "Give only the final answer, without explanation.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def get_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def intersects(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def infer_keep_label_id(model) -> int:
    id2label = getattr(model.config, "id2label", {}) or {}
    for idx, label in id2label.items():
        label_l = str(label).lower()
        if "keep" in label_l or "preserve" in label_l:
            return int(idx)
    return 1


@torch.no_grad()
def score_words_with_compressor(compressor_model, compressor_tokenizer, text: str, device, max_length=512):
    keep_label_id = infer_keep_label_id(compressor_model)

    encoded = compressor_tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = compressor_model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=-1)[:, keep_label_id].detach().cpu().tolist()

    words: List[WordItem] = []

    for i, (w_start, w_end, word) in enumerate(get_word_spans(text)):
        token_scores = []

        for (t_start, t_end), score in zip(offsets, probs):
            if t_start == 0 and t_end == 0:
                continue
            if intersects(w_start, w_end, t_start, t_end):
                token_scores.append(score)

        p_keep = max(token_scores) if token_scores else 0.0

        words.append(
            WordItem(
                index=i,
                text=word,
                start=w_start,
                end=w_end,
                p_keep=float(p_keep),
                final_score=float(p_keep),
            )
        )

    return words


def every_second_word(text: str) -> str:
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i % 2 == 0)


def random_keep(text: str, keep_ratio: float, seed: int) -> str:
    words = text.split()
    if not words:
        return text

    rng = random.Random(seed)
    k = max(1, round(len(words) * keep_ratio))
    keep_indices = set(rng.sample(range(len(words)), k))
    return " ".join(w for i, w in enumerate(words) if i in keep_indices)


def select_topk(words: List[WordItem], keep_ratio: float, score_attr: str = "p_keep") -> str:
    if not words:
        return ""

    k = max(1, round(len(words) * keep_ratio))

    sorted_words = sorted(words, key=lambda w: getattr(w, score_attr), reverse=True)
    keep = set(w.index for w in sorted_words[:k])

    return " ".join(w.text for w in sorted(words, key=lambda w: w.index) if w.index in keep)


def mark_features(words: List[WordItem], text: str, nlp):
    doc = nlp(text)

    ent_spans = []
    for ent in doc.ents:
        ent_spans.append((ent.start_char, ent.end_char, ent.label_))

    noun_root_spans = set()
    for chunk in doc.noun_chunks:
        root = chunk.root
        noun_root_spans.add((root.idx, root.idx + len(root.text)))

    token_features = []

    for tok in doc:
        token_features.append(
            {
                "start": tok.idx,
                "end": tok.idx + len(tok.text),
                "lemma": normalize_word(tok.lemma_ or tok.text),
                "pos": tok.pos_,
                "is_stop": tok.is_stop,
            }
        )

    for w in words:
        low = normalize_word(w.text)

        w.is_number = bool(re.search(r"\d", w.text))
        w.is_negation = low in NEGATIONS or low.endswith("n't")

        for feat in token_features:
            if intersects(w.start, w.end, feat["start"], feat["end"]):
                w.lemma = feat["lemma"] or low
                w.is_verb = w.is_verb or feat["pos"] == "VERB"
                w.is_aux = w.is_aux or feat["pos"] == "AUX"
                w.is_stopword = w.is_stopword or feat["is_stop"]

        if not w.lemma:
            w.lemma = low

        for ent_start, ent_end, ent_label in ent_spans:
            if intersects(w.start, w.end, ent_start, ent_end):
                w.is_entity = True
                if ent_label in {"DATE", "TIME"}:
                    w.is_date = True

        for root_start, root_end in noun_root_spans:
            if intersects(w.start, w.end, root_start, root_end):
                w.is_noun_chunk_root = True


def compute_redundancy(words: List[WordItem]):
    counts = {}
    seen = {}

    for w in words:
        if not w.lemma:
            continue
        counts[w.lemma] = counts.get(w.lemma, 0) + 1

    for w in words:
        if not w.lemma:
            w.redundancy = 0.0
            continue

        c = counts.get(w.lemma, 1)
        prev = seen.get(w.lemma, 0)

        if c <= 1:
            w.redundancy = 0.0
        else:
            w.redundancy = prev / max(1, c - 1)

        seen[w.lemma] = prev + 1

        if w.is_number or w.is_date or w.is_negation:
            w.redundancy = 0.0


def apply_stat_hybrid_scores(words: List[WordItem], args):
    for w in words:
        score = w.p_keep

        score += args.verb_bonus * float(w.is_verb or w.is_aux)
        score += args.noun_chunk_root_bonus * float(w.is_noun_chunk_root)
        score += args.entity_bonus * float(w.is_entity)
        score += args.number_bonus * float(w.is_number)
        score += args.date_bonus * float(w.is_date)
        score += args.negation_bonus * float(w.is_negation)

        score -= args.redundancy_penalty * w.redundancy
        score -= args.stopword_penalty * float(w.is_stopword)

        w.final_score = score


def compress_text(method, question, args, compressor_model, compressor_tokenizer, compressor_device, nlp, row_seed):
    if method == "none":
        return question

    if method == "every_second_word":
        return every_second_word(question)

    if method == "random_keep":
        return random_keep(question, args.keep_ratio, seed=row_seed)

    words = score_words_with_compressor(
        compressor_model=compressor_model,
        compressor_tokenizer=compressor_tokenizer,
        text=question,
        device=compressor_device,
    )

    if method == "model_keep_ratio":
        return select_topk(words, args.keep_ratio, score_attr="p_keep")

    if method == "stat_hybrid":
        mark_features(words, question, nlp)
        compute_redundancy(words)
        apply_stat_hybrid_scores(words, args)
        return select_topk(words, args.keep_ratio, score_attr="final_score")

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def generate_answer(eval_model, eval_tokenizer, prompt: str, max_new_tokens: int, max_input_tokens: int) -> str:
    if getattr(eval_tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = eval_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(eval_model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        encoded = eval_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(eval_model.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    output = eval_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=eval_tokenizer.eos_token_id,
    )

    generated = output[0][input_ids.shape[-1]:]
    text = eval_tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def main():
    args = parse_args()

    random.seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print("Loading evaluator:", args.eval_model)
    eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        args.eval_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    eval_model.eval()

    needs_compressor = any(m in {"model_keep_ratio", "stat_hybrid"} for m in methods)

    compressor_model = None
    compressor_tokenizer = None
    compressor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if needs_compressor:
        print("Loading compressor:", args.compressor_model)
        compressor_tokenizer = AutoTokenizer.from_pretrained(args.compressor_model, use_fast=True)
        compressor_model = AutoModelForTokenClassification.from_pretrained(args.compressor_model)
        compressor_model.to(compressor_device)
        compressor_model.eval()

    nlp = None
    if "stat_hybrid" in methods:
        import spacy
        nlp = spacy.load(args.spacy_model)

    if args.tasks == "auto":
        configs = get_dataset_config_names(args.dataset_name)
        tasks = configs[:3]
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("Tasks:", tasks)
    print("Methods:", methods)

    rows_out = []

    for task in tasks:
        print(f"\nLoading task: {task}")
        ds = load_dataset(args.dataset_name, task, split=args.split)

        indices = list(range(len(ds)))
        random.Random(args.seed).shuffle(indices)
        indices = indices[: args.examples_per_task]

        for local_i, idx in enumerate(indices):
            row = ds[idx]
            question, target = get_text_and_target(row)

            for method in methods:
                compressed_question = compress_text(
                    method=method,
                    question=question,
                    args=args,
                    compressor_model=compressor_model,
                    compressor_tokenizer=compressor_tokenizer,
                    compressor_device=compressor_device,
                    nlp=nlp,
                    row_seed=args.seed + idx,
                )

                prompt = build_prompt(compressed_question)
                pred = generate_answer(
                    eval_model=eval_model,
                    eval_tokenizer=eval_tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                )

                correct = is_correct(pred, target)

                original_words = len(question.split())
                compressed_words = len(compressed_question.split())
                keep_ratio_actual = compressed_words / original_words if original_words else 1.0

                result = {
                    "task": task,
                    "idx": idx,
                    "method": method,
                    "target": target,
                    "prediction": pred,
                    "correct": int(correct),
                    "original_words": original_words,
                    "compressed_words": compressed_words,
                    "keep_ratio_actual": keep_ratio_actual,
                    "question": question,
                    "compressed_question": compressed_question,
                }

                rows_out.append(result)

                print(
                    f"{task} #{idx} | {method} | correct={int(correct)} | "
                    f"keep={keep_ratio_actual:.3f} | pred={pred[:80]!r} | target={target!r}"
                )

    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print("\n=== Summary ===")
    grouped = {}

    for r in rows_out:
        key = r["method"]
        grouped.setdefault(key, []).append(r)

    for method, rs in grouped.items():
        acc = sum(r["correct"] for r in rs) / len(rs)
        avg_keep = sum(r["keep_ratio_actual"] for r in rs) / len(rs)
        print(f"{method}: accuracy={acc:.4f}, avg_keep_ratio={avg_keep:.4f}, n={len(rs)}")

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
