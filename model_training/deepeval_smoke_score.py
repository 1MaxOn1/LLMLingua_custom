#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--judge_model", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    metric = GEval(
        name="BBH Answer Correctness",
        criteria=(
            "Determine whether the actual output gives the same final answer as the expected output. "
            "Ignore formatting differences, explanations, capitalization, punctuation, and parentheses. "
            "For multiple-choice answers, treat '(A)' and 'A' as equivalent. "
            "For boolean answers, treat 'true' and 'false' literally. "
            "Return 1 only if the final answer is equivalent, otherwise return 0."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=args.judge_model,
        threshold=0.5,
        strict_mode=True,
        async_mode=False,
    )

    scores = []
    reasons = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="deepeval-smoke", dynamic_ncols=True):
        test_case = LLMTestCase(
            input=str(row["compressed_question"]),
            actual_output=str(row["prediction"]),
            expected_output=str(row["target"]),
        )

        try:
            metric.measure(test_case)
            scores.append(float(metric.score))
            reasons.append(str(getattr(metric, "reason", "")))
        except Exception as e:
            scores.append(None)
            reasons.append(f"DEEPEVAL_ERROR: {e}")

    df["deepeval_score"] = scores
    df["deepeval_pass"] = df["deepeval_score"].fillna(0) >= 0.5
    df["deepeval_reason"] = reasons

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== Smoke comparison ===")
    print(df.groupby("method").agg(
        exact_accuracy=("correct", "mean"),
        deepeval_accuracy=("deepeval_pass", "mean"),
        avg_deepeval_score=("deepeval_score", "mean"),
        n=("method", "count"),
    ).to_string())

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
