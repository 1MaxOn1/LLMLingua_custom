#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--text_key", default="origin")
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def normalize_word(x: str) -> str:
    return re.sub(r"^\W+|\W+$", "", x.lower())


def main():
    args = parse_args()

    import spacy
    nlp = spacy.load(args.spacy_model, disable=["ner"])

    data = torch.load(args.data_path, map_location="cpu")
    texts = data[args.text_key]

    if args.max_docs is not None:
        texts = texts[: args.max_docs]

    df = Counter()
    num_docs = 0

    for doc in tqdm(nlp.pipe(texts, batch_size=args.batch_size), total=len(texts), desc="Building IDF"):
        lemmas = set()

        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue

            lemma = normalize_word(tok.lemma_ or tok.text)
            if not lemma:
                continue

            lemmas.add(lemma)

        for lemma in lemmas:
            df[lemma] += 1

        num_docs += 1

    idf = {}
    for lemma, count in df.items():
        idf[lemma] = math.log((num_docs + 1) / (count + 1)) + 1.0

    if idf:
        min_idf = min(idf.values())
        max_idf = max(idf.values())
    else:
        min_idf = 0.0
        max_idf = 1.0

    idf_norm = {}
    denom = max(max_idf - min_idf, 1e-12)

    for lemma, value in idf.items():
        idf_norm[lemma] = (value - min_idf) / denom

    out = {
        "num_docs": num_docs,
        "min_idf": min_idf,
        "max_idf": max_idf,
        "idf_norm": idf_norm,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"Saved IDF stats to: {output_path}")
    print(f"Documents: {num_docs}")
    print(f"Lemmas: {len(idf_norm)}")


if __name__ == "__main__":
    main()
