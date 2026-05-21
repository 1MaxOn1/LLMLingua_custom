import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Format MeetingBank dataset for LLMLingua-style data collection.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="huuuyeah/meetingbank",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="data_preparation/src/results/meetingbank/origin",
        help="Directory where formatted MeetingBank JSON files will be saved.",
    )

    # Keep old typo by default because your existing pipeline uses 'formated'.
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="formated",
        choices=["formated", "formatted"],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name)

    for split in dataset:
        data = []

        for idx, instance in enumerate(dataset[split]):
            if "transcript" not in instance:
                raise KeyError(f"Missing 'transcript' field in split={split}, idx={idx}")

            if "summary" not in instance:
                raise KeyError(f"Missing 'summary' field in split={split}, idx={idx}")

            data.append(
                {
                    "idx": idx,
                    "prompt": instance["transcript"],
                    "summary": instance["summary"],
                }
            )

        save_path = save_dir / f"meetingbank_{split}_{args.filename_suffix}.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Saved {split}: {len(data)} samples -> {save_path}")


if __name__ == "__main__":
    main()