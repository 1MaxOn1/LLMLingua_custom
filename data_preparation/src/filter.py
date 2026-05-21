import argparse
import os
from collections import defaultdict

import numpy as np
import torch


parser = argparse.ArgumentParser(description="Filter annotated data")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--save_filtered_path", type=str, default=None)
parser.add_argument("--percentile", type=float, default=90.0)
args = parser.parse_args()


def torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def save_torch(obj, path):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(dict(obj), path)


def subset_by_indices(data, indices):
    out = defaultdict(list)
    indices = set(indices)

    n = len(data["labels"])

    for key, values in data.items():
        if not isinstance(values, list):
            continue

        if len(values) != n:
            print(f"[WARN] skip key={key}, len={len(values)} != {n}")
            continue

        for i in range(n):
            if i in indices:
                out[key].append(values[i])

    return out


def filter_by_metric(data, metric_name, percentile):
    values = np.asarray(data[metric_name], dtype=float)

    if len(values) == 0:
        raise ValueError(f"No values found for metric: {metric_name}")

    if np.isnan(values).any():
        raise ValueError(f"NaN found in metric: {metric_name}")

    threshold = np.percentile(values, percentile)

    kept_indices = []
    filtered_indices = []

    for i, value in enumerate(values):
        # Strictly greater, not >=.
        # This avoids deleting everything when threshold equals a common value like 0.0.
        if value > threshold:
            filtered_indices.append(i)
        else:
            kept_indices.append(i)

    kept = subset_by_indices(data, kept_indices)
    filtered = subset_by_indices(data, filtered_indices)

    return kept, filtered, threshold


res_pt = torch_load(args.load_path)

required_keys = [
    "labels",
    "origin",
    "comp",
    "retrieval",
    "variation_rate",
    "alignment_gap",
]

for key in required_keys:
    if key not in res_pt:
        raise KeyError(f"Missing required key in loaded data: {key}")

num_samples = len(res_pt["labels"])
print("before filtering:", num_samples)

if num_samples == 0:
    raise ValueError("Loaded data is empty.")

# Stage 1: filter high variation_rate.
kept_after_vr, filtered_vr, vr_threshold = filter_by_metric(
    res_pt,
    metric_name="variation_rate",
    percentile=args.percentile,
)

print("variation_rate threshold:", vr_threshold)
print("after variation_rate filtering:", len(kept_after_vr["labels"]))
print("filtered by variation_rate:", len(filtered_vr["labels"]))

if len(kept_after_vr["labels"]) == 0:
    raise ValueError("All samples were filtered out after variation_rate filtering.")

# Stage 2: filter high alignment_gap among remaining samples.
kept_final, filtered_ag, ag_threshold = filter_by_metric(
    kept_after_vr,
    metric_name="alignment_gap",
    percentile=args.percentile,
)

print("alignment_gap threshold:", ag_threshold)
print("after alignment_gap filtering:", len(kept_final["labels"]))
print("filtered by alignment_gap:", len(filtered_ag["labels"]))

# Merge rejected samples from both stages.
filtered_all = defaultdict(list)
for source in [filtered_vr, filtered_ag]:
    for key, values in source.items():
        filtered_all[key].extend(values)

print("filtered total:", len(filtered_all["labels"]))
print("kept final:", len(kept_final["labels"]))

save_torch(kept_final, args.save_path)
print("saved kept data to:", args.save_path)

if args.save_filtered_path is not None:
    save_torch(filtered_all, args.save_filtered_path)
    print("saved filtered data to:", args.save_filtered_path)