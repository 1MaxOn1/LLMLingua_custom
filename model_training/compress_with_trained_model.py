#!/usr/bin/env python3

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


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
    p_keep: float
    final_score: float
    is_stopword: bool = False
    is_entity: bool = False
    is_number: bool = False
    is_negation: bool = False
    is_proper_noun: bool = False
    protected: bool = False
    selected: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress text using LLMLingua2-style token classification model."
    )

    parser.add_argument("--model_path", required=True, help="Local model path or HF model id.")
    parser.add_argument("--text", default=None, help="Input text.")
    parser.add_argument("--text_file", default=None, help="Path to input text file.")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--keep_label_id", type=int, default=None)

    parser.add_argument(
        "--mode",
        choices=["threshold", "keep_ratio", "hybrid"],
        default="threshold",
        help=(
            "threshold: keep p_keep >= threshold; "
            "keep_ratio: keep top-K by p_keep; "
            "hybrid: keep top-K by adjusted final_score with linguistic constraints."
        ),
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--keep_ratio", type=float, default=None)

    parser.add_argument("--protect_entities", action="store_true")
    parser.add_argument("--protect_numbers", action="store_true")
    parser.add_argument("--protect_negations", action="store_true")
    parser.add_argument("--protect_proper_nouns", action="store_true")

    parser.add_argument(
        "--stopword_penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from p_keep for stopwords in hybrid mode.",
    )
    parser.add_argument(
        "--entity_bonus",
        type=float,
        default=0.0,
        help="Bonus added to named entities in hybrid mode.",
    )
    parser.add_argument(
        "--number_bonus",
        type=float,
        default=0.0,
        help="Bonus added to numbers in hybrid mode.",
    )
    parser.add_argument(
        "--negation_bonus",
        type=float,
        default=0.0,
        help="Bonus added to negations in hybrid mode.",
    )

    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--show_scores", action="store_true")

    return parser.parse_args()


def load_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError("Provide either --text or --text_file.")


def infer_keep_label_id(model, explicit_id=None) -> int:
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


def intersects(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def try_load_spacy(model_name: str):
    try:
        import spacy
        return spacy.load(model_name)
    except Exception as e:
        print(f"[warning] spaCy model is unavailable: {e}")
        print("[warning] Entity/proper-noun/stopword features will be limited.")
        return None


def mark_linguistic_features(words: List[WordItem], text: str, nlp) -> None:
    if nlp is None:
        for w in words:
            low = w.text.lower().strip(".,;:!?()[]{}\"'")
            w.is_number = bool(re.search(r"\d", w.text))
            w.is_negation = low in NEGATIONS or low.endswith("n't")
        return

    doc = nlp(text)

    entity_spans = [(ent.start_char, ent.end_char) for ent in doc.ents]

    token_features = []
    for tok in doc:
        token_features.append(
            {
                "start": tok.idx,
                "end": tok.idx + len(tok.text),
                "is_stop": tok.is_stop,
                "is_proper": tok.pos_ == "PROPN",
            }
        )

    for w in words:
        low = w.text.lower().strip(".,;:!?()[]{}\"'")

        w.is_number = bool(re.search(r"\d", w.text))
        w.is_negation = low in NEGATIONS or low.endswith("n't")

        for ent_start, ent_end in entity_spans:
            if intersects(w.start, w.end, ent_start, ent_end):
                w.is_entity = True
                break

        for feat in token_features:
            if intersects(w.start, w.end, feat["start"], feat["end"]):
                w.is_stopword = w.is_stopword or feat["is_stop"]
                w.is_proper_noun = w.is_proper_noun or feat["is_proper"]


@torch.no_grad()
def score_words(model, tokenizer, text: str, device: torch.device, keep_label_id: int, max_length: int):
    encoded_len = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    if encoded_len > max_length:
        print(
            f"[warning] Input has {encoded_len} tokens, max_length={max_length}. "
            f"This script truncates long inputs. For full evaluation, add chunking."
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

    word_spans = get_word_spans(text)
    words: List[WordItem] = []

    for i, (w_start, w_end, word) in enumerate(word_spans):
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


def apply_hybrid_scores(words: List[WordItem], args) -> None:
    for w in words:
        score = w.p_keep

        if w.is_stopword:
            score -= args.stopword_penalty

        if w.is_entity:
            score += args.entity_bonus

        if w.is_number:
            score += args.number_bonus

        if w.is_negation:
            score += args.negation_bonus

        if args.protect_entities and w.is_entity:
            w.protected = True

        if args.protect_numbers and w.is_number:
            w.protected = True

        if args.protect_negations and w.is_negation:
            w.protected = True

        if args.protect_proper_nouns and w.is_proper_noun:
            w.protected = True

        w.final_score = score


def select_words(words: List[WordItem], args) -> None:
    for w in words:
        w.selected = False

    if args.mode == "threshold":
        for w in words:
            w.selected = w.p_keep >= args.threshold
        return

    if args.keep_ratio is None:
        raise ValueError("--keep_ratio is required for mode=keep_ratio or mode=hybrid.")

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("--keep_ratio must be in (0, 1].")

    target_keep = max(1, int(round(len(words) * args.keep_ratio)))

    protected = [w for w in words if w.protected]
    for w in protected:
        w.selected = True

    remaining_budget = target_keep - len(protected)

    if remaining_budget <= 0:
        return

    score_attr = "p_keep" if args.mode == "keep_ratio" else "final_score"

    candidates = [w for w in words if not w.selected]
    candidates.sort(key=lambda x: getattr(x, score_attr), reverse=True)

    for w in candidates[:remaining_budget]:
        w.selected = True


def detokenize_words(words: List[WordItem]) -> str:
    selected = [w.text for w in sorted(words, key=lambda x: x.index) if w.selected]
    return " ".join(selected)


def print_results(text: str, words: List[WordItem], args) -> None:
    compressed = detokenize_words(words)

    original_words = len(words)
    kept_words = sum(1 for w in words if w.selected)

    keep_ratio = kept_words / original_words if original_words else 0.0
    compression_ratio = original_words / kept_words if kept_words else math.inf

    print("\n=== Original ===")
    print(text)

    print("\n=== Compressed ===")
    print(compressed)

    print("\n=== Stats ===")
    print(f"mode: {args.mode}")
    print(f"threshold: {args.threshold}")
    print(f"keep_ratio_target: {args.keep_ratio}")
    print(f"original_words: {original_words}")
    print(f"kept_words: {kept_words}")
    print(f"actual_keep_ratio: {keep_ratio:.4f}")
    print(f"compression_ratio_words: {compression_ratio:.4f}")

    protected_count = sum(1 for w in words if w.protected)
    print(f"protected_words: {protected_count}")

    if args.show_scores:
        print("\n=== Word scores ===")
        print("selected\tprotected\tp_keep\tfinal_score\tfeatures\tword")
        for w in words:
            features = []
            if w.is_stopword:
                features.append("STOP")
            if w.is_entity:
                features.append("ENT")
            if w.is_number:
                features.append("NUM")
            if w.is_negation:
                features.append("NEG")
            if w.is_proper_noun:
                features.append("PROPN")

            print(
                f"{int(w.selected)}\t"
                f"{int(w.protected)}\t"
                f"{w.p_keep:.4f}\t"
                f"{w.final_score:.4f}\t"
                f"{','.join(features)}\t"
                f"{w.text}"
            )


def main():
    args = parse_args()
    text = load_text(args)

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

    nlp = None
    if args.mode == "hybrid" or any(
        [
            args.protect_entities,
            args.protect_numbers,
            args.protect_negations,
            args.protect_proper_nouns,
            args.stopword_penalty > 0,
            args.entity_bonus > 0,
            args.number_bonus > 0,
            args.negation_bonus > 0,
        ]
    ):
        nlp = try_load_spacy(args.spacy_model)

    mark_linguistic_features(words, text, nlp)
    apply_hybrid_scores(words, args)
    select_words(words, args)
    print_results(text, words, args)


if __name__ == "__main__":
    main()
