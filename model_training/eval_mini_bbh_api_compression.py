#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
from openai import OpenAI
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm.auto import tqdm


NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "n't",
}

REASONING_TERMS = {
    "above", "below", "before", "after", "left", "right",
    "less", "more", "than", "most", "least",
    "expensive", "cheaper", "costlier",
    "finished", "finish", "finishes",
    "older", "younger", "taller", "shorter", "heavier", "lighter",
    "first", "second", "third", "fourth", "fifth", "last",
    "true", "false",
    "if", "then", "else",
    "and", "or", "not", "no", "never",
    "all", "some", "none", "every", "each", "either", "neither",
    "same", "different",
    "yes", "only", "always", "sometimes",
}

METHOD_ALIASES = {
    "model_keep_ratio": "finetuned_llmlingua2",
    "stat_hybrid": "finetuned_stat_hybrid",
    "reasoning_safe_hybrid": "finetuned_reasoning_safe",
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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_name", default="lukaemon/bbh")
    p.add_argument(
        "--tasks",
        default="boolean_expressions,date_understanding,logical_deduction_three_objects",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--examples_per_task", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--api_model", required=True)
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--api_base_url_env", default="OPENAI_BASE_URL")
    p.add_argument("--api_temperature", type=float, default=0.0)
    p.add_argument("--api_sleep", type=float, default=0.0)
    p.add_argument("--api_max_retries", type=int, default=3)

    p.add_argument(
        "--compressor_model",
        required=True,
        help="Your fine-tuned LLMLingua2 model path.",
    )
    p.add_argument(
        "--original_compressor_model",
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        help="Original Microsoft LLMLingua2 model.",
    )

    p.add_argument(
        "--methods",
        default=(
            "none,every_second_word,random_keep,"
            "original_llmlingua2,finetuned_llmlingua2,"
            "finetuned_stat_hybrid,finetuned_reasoning_safe"
        ),
    )
    p.add_argument("--keep_ratio", type=float, default=0.6)
    p.add_argument("--max_new_tokens", type=int, default=32)

    p.add_argument("--spacy_model", default="en_core_web_sm")

    p.add_argument("--verb_bonus", type=float, default=0.08)
    p.add_argument("--noun_chunk_root_bonus", type=float, default=0.08)
    p.add_argument("--entity_bonus", type=float, default=0.15)
    p.add_argument("--number_bonus", type=float, default=0.20)
    p.add_argument("--date_bonus", type=float, default=0.20)
    p.add_argument("--negation_bonus", type=float, default=0.30)
    p.add_argument("--redundancy_penalty", type=float, default=0.10)
    p.add_argument("--stopword_penalty", type=float, default=0.00)
    p.add_argument("--reasoning_bonus", type=float, default=0.35)
    p.add_argument("--option_marker_bonus", type=float, default=0.50)

    p.add_argument("--progress_only", action="store_true", help="Show progress bar only; suppress per-example logs.")
    p.add_argument("--output_csv", default="results/eval/mini_bbh_api_compression_results.csv")
    p.add_argument("--output_jsonl", default="results/eval/mini_bbh_api_compression_results.jsonl")

    return p.parse_args()


def canonical_method(method: str) -> str:
    method = method.strip()
    return METHOD_ALIASES.get(method, method)


def make_openai_client(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env}.")

    base_url = os.environ.get(args.api_base_url_env)

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def normalize_word(x: str) -> str:
    return re.sub(r"^\W+|\W+$", "", str(x).lower())


def normalize_answer(x: str) -> str:
    x = str(x).strip().lower()
    x = x.split("\n")[0].strip()
    x = re.sub(r"^(answer|final answer|the answer is)\s*[:\-]?\s*", "", x).strip()
    x = x.replace("(", "").replace(")", "")
    x = x.strip(" .,:;!?\n\t\"'")
    return x


def is_correct(pred: str, target: str) -> bool:
    p = normalize_answer(pred)
    t = normalize_answer(target)

    if p == t:
        return True

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
def score_words_with_compressor(
    compressor_model,
    compressor_tokenizer,
    text: str,
    device,
    max_length: int = 512,
):
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
        if w.lemma:
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


def apply_stat_hybrid_scores(words: List[WordItem], args, reasoning_safe: bool = False):
    for w in words:
        score = w.p_keep
        low = normalize_word(w.text)

        verb_bonus = args.verb_bonus
        noun_bonus = args.noun_chunk_root_bonus
        redundancy_penalty = args.redundancy_penalty

        if reasoning_safe:
            verb_bonus = max(verb_bonus, 0.12)
            noun_bonus = max(noun_bonus, 0.10)
            redundancy_penalty = min(redundancy_penalty, 0.03)

        score += verb_bonus * float(w.is_verb or w.is_aux)
        score += noun_bonus * float(w.is_noun_chunk_root)
        score += args.entity_bonus * float(w.is_entity)
        score += args.number_bonus * float(w.is_number)
        score += args.date_bonus * float(w.is_date)
        score += args.negation_bonus * float(w.is_negation)

        if reasoning_safe:
            if low in REASONING_TERMS:
                score += args.reasoning_bonus

            if re.fullmatch(r"\(?[A-E]\)?", w.text.strip()):
                score += args.option_marker_bonus

            if "second-most" in low or "second" in low:
                score += args.reasoning_bonus

        score -= redundancy_penalty * w.redundancy
        score -= args.stopword_penalty * float(w.is_stopword)

        w.final_score = score


def compress_with_model(
    question: str,
    args,
    model_pack,
    nlp,
    method: str,
):
    compressor_model, compressor_tokenizer, compressor_device = model_pack

    words = score_words_with_compressor(
        compressor_model=compressor_model,
        compressor_tokenizer=compressor_tokenizer,
        text=question,
        device=compressor_device,
    )

    if method in {"original_llmlingua2", "finetuned_llmlingua2"}:
        return select_topk(words, args.keep_ratio, score_attr="p_keep")

    if method in {"finetuned_stat_hybrid", "finetuned_reasoning_safe", "original_reasoning_safe"}:
        mark_features(words, question, nlp)
        compute_redundancy(words)
        apply_stat_hybrid_scores(
            words,
            args,
            reasoning_safe=(method in {"finetuned_reasoning_safe", "original_reasoning_safe"}),
        )
        return select_topk(words, args.keep_ratio, score_attr="final_score")

    raise ValueError(f"Unknown model method: {method}")


def compress_text(
    method: str,
    question: str,
    args,
    finetuned_pack,
    original_pack,
    nlp,
    row_seed: int,
):
    method = canonical_method(method)

    if method == "none":
        return question

    if method == "every_second_word":
        return every_second_word(question)

    if method == "random_keep":
        return random_keep(question, args.keep_ratio, seed=row_seed)

    if method in {"original_llmlingua2", "original_reasoning_safe"}:
        return compress_with_model(question, args, original_pack, nlp, method)

    if method in {"finetuned_llmlingua2", "finetuned_stat_hybrid", "finetuned_reasoning_safe"}:
        return compress_with_model(question, args, finetuned_pack, nlp, method)

    raise ValueError(f"Unknown method: {method}")


def api_generate_answer(client, args, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    last_err = None

    for attempt in range(args.api_max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=args.api_model,
                    messages=messages,
                    temperature=args.api_temperature,
                    max_tokens=args.max_new_tokens,
                )
            except Exception as e:
                if "max_tokens" not in str(e):
                    raise
                resp = client.chat.completions.create(
                    model=args.api_model,
                    messages=messages,
                    temperature=args.api_temperature,
                    max_completion_tokens=args.max_new_tokens,
                )

            text = resp.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 8)
            print(f"[warning] API call failed, retry {attempt + 1}/{args.api_max_retries}: {e}")
            time.sleep(sleep_s)

    raise RuntimeError(f"API failed after retries: {last_err}")


def load_compressor_pack(model_path: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    args = parse_args()

    random.seed(args.seed)

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("--keep_ratio must be in (0, 1].")

    methods = [canonical_method(m) for m in args.methods.split(",") if m.strip()]

    valid_methods = {
        "none",
        "every_second_word",
        "random_keep",
        "original_llmlingua2",
        "original_reasoning_safe",
        "finetuned_llmlingua2",
        "finetuned_stat_hybrid",
        "finetuned_reasoning_safe",
    }

    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown method: {m}. Valid methods: {sorted(valid_methods)}")

    client = make_openai_client(args)

    needs_finetuned = any(
        m in {"finetuned_llmlingua2", "finetuned_stat_hybrid", "finetuned_reasoning_safe"}
        for m in methods
    )
    needs_original = any(m in {"original_llmlingua2", "original_reasoning_safe"} for m in methods)
    needs_spacy = any(m in {"finetuned_stat_hybrid", "finetuned_reasoning_safe", "original_reasoning_safe"} for m in methods)

    compressor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_pack = None
    original_pack = None
    nlp = None

    if needs_finetuned:
        print("Loading fine-tuned compressor:", args.compressor_model)
        finetuned_pack = load_compressor_pack(args.compressor_model, compressor_device)
        print("Fine-tuned compressor device:", compressor_device)

    if needs_original:
        print("Loading original compressor:", args.original_compressor_model)
        original_pack = load_compressor_pack(args.original_compressor_model, compressor_device)
        print("Original compressor device:", compressor_device)

    if needs_spacy:
        import spacy
        nlp = spacy.load(args.spacy_model)

    if args.tasks == "auto":
        configs = get_dataset_config_names(args.dataset_name)
        tasks = configs[:3]
    elif args.tasks == "all":
        tasks = get_dataset_config_names(args.dataset_name)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("Dataset:", args.dataset_name)
    print("Tasks:", tasks)
    print("Methods:", methods)
    print("API evaluator:", args.api_model)

    rows_out = []

    if args.progress_only:
        total_examples = 0
        for task_for_count in tasks:
            ds_for_count = load_dataset(args.dataset_name, task_for_count, split=args.split)
            if args.examples_per_task and args.examples_per_task > 0:
                total_examples += min(args.examples_per_task, len(ds_for_count))
            else:
                total_examples += len(ds_for_count)

        total_steps = total_examples * len(methods)
        pbar = tqdm(total=total_steps, desc='eval', dynamic_ncols=True)
    else:
        pbar = None

    for task in tasks:
        print(f"\nLoading task: {task}")
        ds = load_dataset(args.dataset_name, task, split=args.split)

        indices = list(range(len(ds)))
        random.Random(args.seed).shuffle(indices)

        if args.examples_per_task and args.examples_per_task > 0:
            indices = indices[: args.examples_per_task]

        for idx in indices:
            row = ds[idx]
            question, target = get_text_and_target(row)

            for method in methods:
                compressed_question = compress_text(
                    method=method,
                    question=question,
                    args=args,
                    finetuned_pack=finetuned_pack,
                    original_pack=original_pack,
                    nlp=nlp,
                    row_seed=args.seed + idx,
                )

                prompt = build_prompt(compressed_question)
                pred = api_generate_answer(client, args, prompt)
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

                if args.progress_only:
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            "method": method,
                            "correct": int(correct),
                            "keep": f"{keep_ratio_actual:.3f}",
                        })
                else:
                    print(
                        f"{task} #{idx} | {method} | correct={int(correct)} | "
                        f"keep={keep_ratio_actual:.3f} | pred={pred[:80]!r} | target={target!r}"
                    )

                if args.api_sleep > 0:
                    time.sleep(args.api_sleep)

    if pbar is not None:
        pbar.close()

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
        grouped.setdefault(r["method"], []).append(r)

    for method, rs in grouped.items():
        acc = sum(r["correct"] for r in rs) / len(rs)
        avg_keep = sum(r["keep_ratio_actual"] for r in rs) / len(rs)
        print(f"{method}: accuracy={acc:.4f}, avg_keep_ratio={avg_keep:.4f}, n={len(rs)}")

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
