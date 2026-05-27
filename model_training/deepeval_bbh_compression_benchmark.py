#!/usr/bin/env python3

import argparse
import json
import os
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from types import SimpleNamespace

import torch
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.tasks import BigBenchHardTask

from model_training.eval_mini_bbh_api_compression import (
    compress_text,
    load_compressor_pack,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--api_model", required=True)
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--api_base_url_env", default="OPENAI_BASE_URL")

    p.add_argument(
        "--compressor_model",
        default="results/models/llmlingua2_microsoft_qwen_full_e5_bs16_lr5e6/best",
    )
    p.add_argument(
        "--original_compressor_model",
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    )

    p.add_argument(
        "--methods",
        default=(
            "none,"
            "original_llmlingua2,"
            "original_reasoning_safe,"
            "finetuned_llmlingua2,"
            "finetuned_reasoning_safe"
        ),
    )

    p.add_argument("--keep_ratio", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--tasks", default="all")
    p.add_argument("--n_shots", type=int, default=0)
    p.add_argument("--enable_cot", action="store_true")

    p.add_argument("--spacy_model", default="en_core_web_sm")

    p.add_argument("--output_json", default="results/eval/deepeval_bbh_compression_results.json")

    return p.parse_args()


def parse_tasks(task_arg: str):
    if task_arg == "all":
        return None

    mapping = {task.name.lower(): task for task in BigBenchHardTask}
    tasks = []

    for raw in task_arg.split(","):
        name = raw.strip().lower()
        if not name:
            continue

        if name not in mapping:
            raise ValueError(
                f"Unknown task: {raw}. Available: {', '.join(sorted(mapping.keys()))}"
            )

        tasks.append(mapping[name])

    return tasks


def make_compression_args(args):
    return SimpleNamespace(
        keep_ratio=args.keep_ratio,
        verb_bonus=0.08,
        noun_chunk_root_bonus=0.08,
        entity_bonus=0.15,
        number_bonus=0.20,
        date_bonus=0.20,
        negation_bonus=0.30,
        redundancy_penalty=0.10,
        stopword_penalty=0.00,
        reasoning_bonus=0.35,
        option_marker_bonus=0.50,
    )


class CompressedQwenModel(DeepEvalBaseLLM):
    def __init__(
        self,
        method,
        api_model,
        api_key,
        base_url,
        compression_args,
        finetuned_pack=None,
        original_pack=None,
        nlp=None,
        max_tokens=32,
        temperature=0.0,
    ):
        self.method = method
        self.api_model = api_model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.compression_args = compression_args
        self.finetuned_pack = finetuned_pack
        self.original_pack = original_pack
        self.nlp = nlp
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        compressed_prompt = compress_text(
            method=self.method,
            question=prompt,
            args=self.compression_args,
            finetuned_pack=self.finetuned_pack,
            original_pack=self.original_pack,
            nlp=self.nlp,
            row_seed=42,
        )

        last_error = None

        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": compressed_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                sleep_s = min(2 ** attempt, 30)
                print(
                    f"[warning] API error in method={self.method}, "
                    f"attempt={attempt + 1}/6, sleep={sleep_s}s: {e}",
                    flush=True,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"API failed after retries: {last_error}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt=prompt, schema=schema)

    def get_model_name(self):
        return f"{self.api_model}__{self.method}__kr{self.compression_args.keep_ratio}"


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    base_url = os.environ.get(args.api_base_url_env)

    if not api_key:
        raise RuntimeError(f"Missing API key env: {args.api_key_env}")
    if not base_url:
        raise RuntimeError(f"Missing API base URL env: {args.api_base_url_env}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    needs_finetuned = any(m.startswith("finetuned") for m in methods)
    needs_original = any(m.startswith("original") for m in methods)
    needs_spacy = any("reasoning_safe" in m or "stat_hybrid" in m for m in methods)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    finetuned_pack = None
    original_pack = None
    nlp = None

    if needs_finetuned:
        print(f"Loading fine-tuned compressor: {args.compressor_model}")
        finetuned_pack = load_compressor_pack(args.compressor_model, device)

    if needs_original:
        print(f"Loading original compressor: {args.original_compressor_model}")
        original_pack = load_compressor_pack(args.original_compressor_model, device)

    if needs_spacy:
        import spacy
        print(f"Loading spaCy: {args.spacy_model}")
        nlp = spacy.load(args.spacy_model)

    compression_args = make_compression_args(args)
    tasks = parse_tasks(args.tasks)

    results = {}

    for method in methods:
        print("\n" + "=" * 80)
        print(f"Evaluating method: {method}")
        print("=" * 80)

        benchmark_kwargs = {
            "n_shots": args.n_shots,
            "enable_cot": args.enable_cot,
        }

        if tasks is not None:
            benchmark_kwargs["tasks"] = tasks

        benchmark = BigBenchHard(**benchmark_kwargs)

        model = CompressedQwenModel(
            method=method,
            api_model=args.api_model,
            api_key=api_key,
            base_url=base_url,
            compression_args=compression_args,
            finetuned_pack=finetuned_pack,
            original_pack=original_pack,
            nlp=nlp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        benchmark.evaluate(model=model)

        score = float(benchmark.overall_score)
        print(f"{method}: overall_score={score:.6f}")

        method_result = {
            "method": method,
            "overall_score": score,
            "keep_ratio": args.keep_ratio,
            "api_model": args.api_model,
            "n_shots": args.n_shots,
            "enable_cot": args.enable_cot,
            "tasks": "all" if tasks is None else [t.name for t in tasks],
        }

        # DeepEval versions expose different extra fields. Save them if present.
        for attr in ["task_scores", "predictions", "results"]:
            if hasattr(benchmark, attr):
                value = getattr(benchmark, attr)
                try:
                    json.dumps(value, default=str)
                    method_result[attr] = value
                except Exception:
                    method_result[attr] = str(value)

        results[method] = method_result

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== Final DeepEval BBH Summary ===")
    for method, result in sorted(results.items(), key=lambda x: x[1]["overall_score"], reverse=True):
        print(f"{method}: {result['overall_score']:.6f}")

    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
