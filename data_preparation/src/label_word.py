import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import spacy
import torch
from tqdm import tqdm


BAD_PREFIXES = [
    "The compressed text is:",
    "The text is compressed to:",
    "Compressed text:",
    "Here is the compressed text:",
    "The compressed version is:",
    "Here is the compressed version:",
    "The compression is:",
    "Output:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate word labels for LLMLingua-2-style data collection.")

    parser.add_argument("--load_prompt_from", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=400)

    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--verbose", action="store_true", default=False)

    # Do not spam stdout on huge datasets.
    parser.add_argument("--debug_alignment_gap", type=float, default=0.1)
    parser.add_argument("--max_debug_examples", type=int, default=20)

    # Periodic checkpointing.
    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()

    if not args.save_path.endswith(".json"):
        raise ValueError("--save_path must end with .json; the .pt file will be created automatically.")

    if args.window_size <= 0:
        raise ValueError("--window_size must be positive.")

    return args


def setup_logging(save_path: str) -> logging.Logger:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir or ".", "log.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    return logger


def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def normalize_wrapper_prefix(text: str) -> str:
    """
    Remove common Qwen/LLM assistant wrappers.

    Example:
        "The compressed text is:\\n\\nfoo bar" -> "foo bar"
    """
    text = str(text).strip()

    # Remove markdown code fences if the model produced them.
    text = re.sub(r"^```(?:text|txt|markdown)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    changed = True
    while changed:
        changed = False

        # Remove repeated known prefixes.
        for prefix in BAD_PREFIXES:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                changed = True

        # Remove simple quoted wrapping.
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
            changed = True

    return text


def clean_comp(text: Any) -> str:
    text = "" if text is None else str(text)
    text = normalize_wrapper_prefix(text)

    # Normalize excessive whitespace but preserve sentence boundaries enough for spaCy.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_string(text: Any, nlp) -> List[str]:
    """
    Convert text into normalized word-level tokens.

    Important:
    - skip punctuation and whitespace;
    - use lemmas for alignment robustness;
    - lowercase everything so matching is stable;
    - keep numbers, because MeetingBank contains bill numbers, dates, sections, etc.
    """
    text = "" if text is None else str(text)
    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.strip()

        # Some spaCy versions/models may produce special or empty lemmas.
        if not lemma or lemma == "-PRON-":
            lemma = token.text.strip()

        lemma = lemma.lower()

        if not lemma:
            continue

        tokens.append(lemma)

    return tokens


def is_equal(token1: str, token2: str) -> bool:
    return token1 == token2


def iter_samples(raw_data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Deterministic iteration over either:
    - dict with string ids: {"0": {...}, "1": {...}}
    - list of samples: [{...}, {...}]
    """
    if isinstance(raw_data, dict):
        keys = list(raw_data.keys())

        try:
            keys = sorted(keys, key=lambda x: int(x))
        except ValueError:
            # Fall back to insertion order if keys are not numeric.
            pass

        for key in keys:
            yield str(key), raw_data[key]

    elif isinstance(raw_data, list):
        for idx, sample in enumerate(raw_data):
            yield str(idx), sample

    else:
        raise TypeError("Input JSON must be either a dict or a list.")


def flatten_prompt_pairs(raw_data: Any) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Converts original compressed dataset into chunk-level pairs:
        origin_chunk -> compressed_chunk

    Returns:
        origins: list[str]
        comps: list[str]
        meta: list[dict]
    """
    origins = []
    comps = []
    meta = []

    skipped_mismatched = 0

    for sample_id, sample in iter_samples(raw_data):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample {sample_id} must be a dict, got {type(sample)}.")

        if "prompt_list" in sample and "compressed_prompt_list" in sample:
            prompt_list = sample["prompt_list"]
            compressed_prompt_list = sample["compressed_prompt_list"]

            if len(prompt_list) != len(compressed_prompt_list):
                skipped_mismatched += 1
                print(
                    f"[WARN] sample {sample_id}: len(prompt_list)={len(prompt_list)} != "
                    f"len(compressed_prompt_list)={len(compressed_prompt_list)}. Skipped."
                )
                continue

            original_idx = sample.get("idx", sample_id)

            for chunk_id, (origin_chunk, comp_chunk) in enumerate(zip(prompt_list, compressed_prompt_list)):
                origins.append(str(origin_chunk))
                comps.append(clean_comp(comp_chunk))
                meta.append(
                    {
                        "sample_id": str(sample_id),
                        "original_idx": original_idx,
                        "chunk_id": chunk_id,
                    }
                )

        else:
            if "prompt" not in sample or "compressed_prompt" not in sample:
                raise KeyError(
                    f"Sample {sample_id} must contain either "
                    f"('prompt_list', 'compressed_prompt_list') or ('prompt', 'compressed_prompt')."
                )

            origins.append(str(sample["prompt"]))
            comps.append(clean_comp(sample["compressed_prompt"]))
            meta.append(
                {
                    "sample_id": str(sample_id),
                    "original_idx": sample.get("idx", sample_id),
                    "chunk_id": None,
                }
            )

    if skipped_mismatched > 0:
        print(f"[WARN] skipped mismatched samples: {skipped_mismatched}")

    return origins, comps, meta


def align_tokens(
    origin_tokens: List[str],
    comp_tokens: List[str],
    window_size: int,
    verbose: bool = False,
) -> Tuple[List[bool], int]:
    """
    Greedy local alignment.

    For each compressed token:
    - count whether it appears in origin token set;
    - try to align it around the previous matched position;
    - mark matched origin token as True.

    Returns:
        labels: bool labels for origin tokens
        num_find: number of compressed tokens that exist somewhere in origin
    """
    num_origin_tokens = len(origin_tokens)
    labels = [False] * num_origin_tokens

    origin_tokens_set = set(origin_tokens)
    num_find = 0
    prev_idx = 0

    for token in comp_tokens:
        if token in origin_tokens_set:
            num_find += 1

        for offset in range(window_size):
            # Look forward from current alignment position.
            token_idx = min(prev_idx + offset, num_origin_tokens - 1)

            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True

                # Keep the window from jumping too aggressively.
                if token_idx - prev_idx > window_size // 2:
                    prev_idx += window_size // 2
                else:
                    prev_idx = token_idx

                if verbose:
                    print(
                        "[MATCH-FWD]",
                        token,
                        "token_idx=",
                        token_idx,
                        "prev_idx=",
                        prev_idx,
                        "context=",
                        origin_tokens[max(token_idx - 2, 0): token_idx + 3],
                    )

                break

            # Look backward from current alignment position.
            token_idx = max(prev_idx - offset, 0)

            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                prev_idx = token_idx

                if verbose:
                    print(
                        "[MATCH-BWD]",
                        token,
                        "token_idx=",
                        token_idx,
                        "prev_idx=",
                        prev_idx,
                        "context=",
                        origin_tokens[max(token_idx - 2, 0): token_idx + 3],
                    )

                break

    return labels, num_find


def compute_metrics(
    origin_tokens: List[str],
    comp_tokens: List[str],
    labels: List[bool],
    num_find: int,
) -> Dict[str, float]:
    num_origin_tokens = len(origin_tokens)
    num_comp_tokens = len(comp_tokens)

    comp_rate = num_comp_tokens / num_origin_tokens if num_origin_tokens > 0 else 0.0
    find_rate = num_find / num_comp_tokens if num_comp_tokens > 0 else 0.0

    variation_rate = 1.0 - find_rate
    hitting_rate = num_find / num_origin_tokens if num_origin_tokens > 0 else 0.0
    matching_rate = sum(labels) / len(labels) if labels else 0.0

    # Same diagnostic idea as LLMLingua-2 data collection:
    # if many compressed tokens exist in origin but are not aligned as labels,
    # this gap becomes large.
    alignment_gap = hitting_rate - matching_rate

    return {
        "comp_rate": comp_rate,
        "find_rate": find_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
    }


def save_outputs(save_path: str, res: Dict[str, Any], res_pt: Dict[str, List[Any]]) -> None:
    pt_path = save_path.replace(".json", ".pt")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    torch.save(dict(res_pt), pt_path)


def main():
    args = parse_args()
    logger = setup_logging(args.save_path)
    nlp = load_spacy_model(args.spacy_model)

    with open(args.load_prompt_from, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    origins, comps, meta = flatten_prompt_pairs(raw_data)

    if len(origins) != len(comps):
        raise RuntimeError(f"Internal error: len(origins)={len(origins)} != len(comps)={len(comps)}.")

    if len(origins) == 0:
        raise ValueError("No valid prompt/compressed pairs found.")

    print(f"Loaded chunk pairs: {len(origins)}")
    logger.info(f"Loaded chunk pairs: {len(origins)}")

    res = {}
    res_pt = defaultdict(list)

    processed_sample = 0
    skipped_empty_origin = 0

    metric_sums = defaultdict(float)

    debug_printed = 0

    for chunk_idx, (origin, comp, item_meta) in tqdm(
        enumerate(zip(origins, comps, meta)),
        total=len(origins),
    ):
        origin_tokens = split_string(origin, nlp)
        comp_tokens = split_string(comp, nlp)

        if len(origin_tokens) == 0:
            skipped_empty_origin += 1
            continue

        labels, num_find = align_tokens(
            origin_tokens=origin_tokens,
            comp_tokens=comp_tokens,
            window_size=args.window_size,
            verbose=args.verbose,
        )

        retrieval_tokens = [token for token, label in zip(origin_tokens, labels) if label]
        retrieval = " ".join(retrieval_tokens)

        metrics = compute_metrics(
            origin_tokens=origin_tokens,
            comp_tokens=comp_tokens,
            labels=labels,
            num_find=num_find,
        )

        processed_sample += 1

        for key, value in metrics.items():
            metric_sums[key] += value

        if (
            metrics["alignment_gap"] > args.debug_alignment_gap
            and debug_printed < args.max_debug_examples
        ):
            debug_printed += 1

            debug_text = (
                "\n"
                + "=" * 80
                + f"\n[DEBUG] chunk_idx={chunk_idx}, meta={item_meta}\n"
                + "-" * 80
                + f"\nORIGIN:\n{origin}\n"
                + "-" * 80
                + f"\nCOMP:\n{comp}\n"
                + "-" * 80
                + f"\nRETRIEVAL:\n{retrieval}\n"
                + "-" * 80
                + (
                    "\n"
                    f"comp_rate={metrics['comp_rate']:.4f}, "
                    f"find_rate={metrics['find_rate']:.4f}, "
                    f"variation_rate={metrics['variation_rate']:.4f}, "
                    f"hitting_rate={metrics['hitting_rate']:.4f}, "
                    f"matching_rate={metrics['matching_rate']:.4f}, "
                    f"alignment_gap={metrics['alignment_gap']:.4f}"
                )
                + "\n"
                + "=" * 80
            )

            print(debug_text)
            logger.info(debug_text)

        item = {
            "labels": labels,
            "origin": origin,
            "comp": comp,
            "retrieval": retrieval,
            "origin_tokens": origin_tokens,
            "comp_tokens": comp_tokens,
            "num_find": num_find,
            "sample_id": item_meta["sample_id"],
            "original_idx": item_meta["original_idx"],
            "chunk_id": item_meta["chunk_id"],
            "comp_rate": metrics["comp_rate"],
            "find_rate": metrics["find_rate"],
            "variation_rate": metrics["variation_rate"],
            "hitting_rate": metrics["hitting_rate"],
            "matching_rate": metrics["matching_rate"],
            "alignment_gap": metrics["alignment_gap"],
        }

        res[str(chunk_idx)] = item

        # Keep this format compatible with the next filtering/training stages.
        res_pt["labels"].append(labels)
        res_pt["origin"].append(origin)
        res_pt["comp"].append(comp)
        res_pt["retrieval"].append(retrieval)
        res_pt["origin_tokens"].append(origin_tokens)
        res_pt["comp_tokens"].append(comp_tokens)
        res_pt["num_find"].append(num_find)
        res_pt["sample_id"].append(item_meta["sample_id"])
        res_pt["original_idx"].append(item_meta["original_idx"])
        res_pt["chunk_id"].append(item_meta["chunk_id"])
        res_pt["comp_rate"].append(metrics["comp_rate"])
        res_pt["find_rate"].append(metrics["find_rate"])
        res_pt["variation_rate"].append(metrics["variation_rate"])
        res_pt["hitting_rate"].append(metrics["hitting_rate"])
        res_pt["matching_rate"].append(metrics["matching_rate"])
        res_pt["alignment_gap"].append(metrics["alignment_gap"])

        if args.save_every > 0 and processed_sample % args.save_every == 0:
            save_outputs(args.save_path, res, res_pt)
            logger.info(f"Checkpoint saved after processed_sample={processed_sample}")

    save_outputs(args.save_path, res, res_pt)

    if processed_sample == 0:
        raise ValueError("No samples were processed. Check your input file.")

    metric_avgs = {
        key: value / processed_sample
        for key, value in metric_sums.items()
    }

    print_info = (
        f"window_size: {args.window_size}, "
        f"processed_sample: {processed_sample}, "
        f"skipped_empty_origin: {skipped_empty_origin}, "
        f"comp_rate: {metric_avgs.get('comp_rate', 0.0):.6f}, "
        f"find_rate: {metric_avgs.get('find_rate', 0.0):.6f}, "
        f"hitting_rate: {metric_avgs.get('hitting_rate', 0.0):.6f}, "
        f"retrieval_rate: {metric_avgs.get('matching_rate', 0.0):.6f}, "
        f"variation_rate: {metric_avgs.get('variation_rate', 0.0):.6f}, "
        f"alignment_gap: {metric_avgs.get('alignment_gap', 0.0):.6f}"
    )

    print(print_info)
    logger.info(print_info)


if __name__ == "__main__":
    main()