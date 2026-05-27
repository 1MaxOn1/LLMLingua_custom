#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "n't",
}

BOILERPLATE_PHRASES = [
    "thank you",
    "thanks",
    "you know",
    "i think",
    "i mean",
    "sort of",
    "kind of",
    "as i said",
    "as mentioned",
    "let's move on",
    "can you hear me",
    "good morning",
    "good afternoon",
    "okay",
    "all right",
]


@dataclass
class WordItem:
    index: int
    text: str
    start: int
    end: int

    lemma: str = ""
    p_keep: float = 0.0
    final_score: float = 0.0

    is_stopword: bool = False
    is_entity: bool = False
    is_number: bool = False
    is_date: bool = False
    is_negation: bool = False
    is_verb: bool = False
    is_aux: bool = False
    is_proper_noun: bool = False
    is_noun_chunk_root: bool = False
    is_boilerplate: bool = False

    idf_norm: float = 0.0
    redundancy: float = 0.0

    protected: bool = False
    selected: bool = False


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--text_file", default=None)

    p.add_argument("--idf_path", default=None)
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--keep_label_id", type=int, default=None)

    p.add_argument("--keep_ratio", type=float, required=True)

    p.add_argument("--idf_bonus", type=float, default=0.05)
    p.add_argument("--verb_bonus", type=float, default=0.08)
    p.add_argument("--noun_chunk_root_bonus", type=float, default=0.08)
    p.add_argument("--entity_bonus", type=float, default=0.15)
    p.add_argument("--number_bonus", type=float, default=0.20)
    p.add_argument("--date_bonus", type=float, default=0.20)
    p.add_argument("--negation_bonus", type=float, default=0.30)

    p.add_argument("--redundancy_penalty", type=float, default=0.10)
    p.add_argument("--boilerplate_penalty", type=float, default=0.05)
    p.add_argument("--stopword_penalty", type=float, default=0.00)

    p.add_argument("--protect_entities", action="store_true")
    p.add_argument("--protect_numbers", action="store_true")
    p.add_argument("--protect_dates", action="store_true")
    p.add_argument("--protect_negations", action="store_true")

    p.add_argument("--show_scores", action="store_true")

    return p.parse_args()


def load_text(args):
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --text or --text_file")


def normalize_word(x: str) -> str:
    return re.sub(r"^\W+|\W+$", "", x.lower())


def infer_keep_label_id(model, explicit_id=None):
    if explicit_id is not None:
        return explicit_id

    id2label = getattr(model.config, "id2label", {}) or {}
    for idx, label in id2label.items():
        label_l = str(label).lower()
        if "keep" in label_l or "preserve" in label_l:
            return int(idx)

    return 1


def get_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def intersects(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def load_idf(path):
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("idf_norm", {})


def mark_boilerplate(words: List[WordItem], text: str):
    low = text.lower()

    for phrase in BOILERPLATE_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        for m in re.finditer(pattern, low):
            for w in words:
                if intersects(w.start, w.end, m.start(), m.end()):
                    w.is_boilerplate = True


def mark_features(words: List[WordItem], text: str, nlp, idf_norm):
    doc = nlp(text)

    ent_spans = []
    date_spans = []

    for ent in doc.ents:
        ent_spans.append((ent.start_char, ent.end_char, ent.label_))
        if ent.label_ in {"DATE", "TIME"}:
            date_spans.append((ent.start_char, ent.end_char))

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
                "is_stop": tok.is_stop,
                "pos": tok.pos_,
            }
        )

    for w in words:
        raw_low = normalize_word(w.text)
        w.is_number = bool(re.search(r"\d", w.text))
        w.is_negation = raw_low in NEGATIONS or raw_low.endswith("n't")

        for feat in token_features:
            if intersects(w.start, w.end, feat["start"], feat["end"]):
                w.lemma = feat["lemma"] or raw_low
                w.is_stopword = w.is_stopword or feat["is_stop"]
                w.is_verb = w.is_verb or feat["pos"] == "VERB"
                w.is_aux = w.is_aux or feat["pos"] == "AUX"
                w.is_proper_noun = w.is_proper_noun or feat["pos"] == "PROPN"

        if not w.lemma:
            w.lemma = raw_low

        for ent_start, ent_end, ent_label in ent_spans:
            if intersects(w.start, w.end, ent_start, ent_end):
                w.is_entity = True
                if ent_label in {"DATE", "TIME"}:
                    w.is_date = True

        for root_start, root_end in noun_root_spans:
            if intersects(w.start, w.end, root_start, root_end):
                w.is_noun_chunk_root = True

        w.idf_norm = float(idf_norm.get(w.lemma, 0.0))

    mark_boilerplate(words, text)


def compute_redundancy(words: List[WordItem]):
    counts = Counter(w.lemma for w in words if w.lemma)
    seen = defaultdict(int)

    for w in words:
        c = counts.get(w.lemma, 1)

        if c <= 1:
            w.redundancy = 0.0
        else:
            # First mention gets 0, later repeated mentions get larger penalty.
            w.redundancy = seen[w.lemma] / max(1, c - 1)

        seen[w.lemma] += 1

        # Do not punish factual/safety-critical tokens strongly.
        if w.is_number or w.is_date or w.is_negation:
            w.redundancy = 0.0


@torch.no_grad()
def score_words(model, tokenizer, text, device, keep_label_id, max_length):
    encoded_len = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    if encoded_len > max_length:
        print(
            f"[warning] Input has {encoded_len} tokens, max_length={max_length}. "
            "This script truncates long inputs."
        )

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=-1)[:, keep_label_id].detach().cpu().tolist()

    words = []

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


def apply_final_score(words: List[WordItem], args):
    for w in words:
        score = w.p_keep

        score += args.idf_bonus * w.idf_norm
        score += args.verb_bonus * float(w.is_verb or w.is_aux)
        score += args.noun_chunk_root_bonus * float(w.is_noun_chunk_root)
        score += args.entity_bonus * float(w.is_entity)
        score += args.number_bonus * float(w.is_number)
        score += args.date_bonus * float(w.is_date)
        score += args.negation_bonus * float(w.is_negation)

        score -= args.redundancy_penalty * w.redundancy
        score -= args.boilerplate_penalty * float(w.is_boilerplate)
        score -= args.stopword_penalty * float(w.is_stopword)

        if args.protect_entities and w.is_entity:
            w.protected = True
        if args.protect_numbers and w.is_number:
            w.protected = True
        if args.protect_dates and w.is_date:
            w.protected = True
        if args.protect_negations and w.is_negation:
            w.protected = True

        w.final_score = score


def select_topk(words: List[WordItem], keep_ratio: float):
    for w in words:
        w.selected = False

    target_keep = max(1, int(round(len(words) * keep_ratio)))

    protected = [w for w in words if w.protected]
    for w in protected:
        w.selected = True

    remaining_budget = target_keep - len(protected)

    if remaining_budget <= 0:
        return

    candidates = [w for w in words if not w.selected]
    candidates.sort(key=lambda x: x.final_score, reverse=True)

    for w in candidates[:remaining_budget]:
        w.selected = True


def detokenize(words: List[WordItem]):
    return " ".join(w.text for w in sorted(words, key=lambda x: x.index) if w.selected)


def print_results(text, words, args):
    compressed = detokenize(words)
    original_words = len(words)
    kept_words = sum(w.selected for w in words)

    print("\n=== Original ===")
    print(text)

    print("\n=== Compressed ===")
    print(compressed)

    print("\n=== Stats ===")
    print(f"keep_ratio_target: {args.keep_ratio}")
    print(f"original_words: {original_words}")
    print(f"kept_words: {kept_words}")
    print(f"actual_keep_ratio: {kept_words / original_words:.4f}")
    print(f"compression_ratio_words: {original_words / kept_words:.4f}")
    print(f"protected_words: {sum(w.protected for w in words)}")

    if args.show_scores:
        print("\n=== Word scores ===")
        print("sel\tprot\tp_keep\tfinal\tidf\tred\tfeatures\tword")
        for w in words:
            feats = []
            if w.is_stopword:
                feats.append("STOP")
            if w.is_entity:
                feats.append("ENT")
            if w.is_number:
                feats.append("NUM")
            if w.is_date:
                feats.append("DATE")
            if w.is_negation:
                feats.append("NEG")
            if w.is_verb:
                feats.append("VERB")
            if w.is_aux:
                feats.append("AUX")
            if w.is_noun_chunk_root:
                feats.append("NCH_ROOT")
            if w.is_boilerplate:
                feats.append("BOILER")

            print(
                f"{int(w.selected)}\t"
                f"{int(w.protected)}\t"
                f"{w.p_keep:.4f}\t"
                f"{w.final_score:.4f}\t"
                f"{w.idf_norm:.4f}\t"
                f"{w.redundancy:.4f}\t"
                f"{','.join(feats)}\t"
                f"{w.text}"
            )


def main():
    args = parse_args()
    text = load_text(args)

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("--keep_ratio must be in (0, 1].")

    import spacy
    nlp = spacy.load(args.spacy_model)

    idf_norm = load_idf(args.idf_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    keep_label_id = infer_keep_label_id(model, args.keep_label_id)
    print(f"Using keep_label_id={keep_label_id}")
    print(f"Model labels: {getattr(model.config, 'id2label', None)}")

    words = score_words(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        keep_label_id=keep_label_id,
        max_length=args.max_length,
    )

    mark_features(words, text, nlp, idf_norm)
    compute_redundancy(words)
    apply_final_score(words, args)
    select_topk(words, args.keep_ratio)
    print_results(text, words, args)


if __name__ == "__main__":
    main()
