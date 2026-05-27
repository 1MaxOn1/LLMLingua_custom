import argparse
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import (
    CompressionTokenDataset,
    compute_token_metrics,
    make_collator,
    set_seed,
    split_dataset,
    torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RoBERTa token classifier for LLMLingua-style compression."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="Base encoder model, e.g. roberta-base, xlm-roberta-base, bert-base-uncased.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data_preparation/src/results/qwen25_comp/annotation/qwen_labeled_kept.pt",
        help="Path to filtered labeled .pt file.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="results/models/roberta_base_qwen_meetingbank",
        help="Directory to save trained model and tokenizer.",
    )

    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only first N samples for smoke-test/debug.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Keep 0 on Windows unless you know what you are doing.",
    )

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision. Only works on CUDA.",
    )

    parser.add_argument(
        "--label_first_subtoken_only",
        action="store_true",
        help="If set, compute loss only on first subtoken of each word.",
    )

    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")

    return args


def load_tokenizer(model_name: str):
    """
    RoBERTa-family tokenizers need add_prefix_space=True when using
    is_split_into_words=True.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

    if not tokenizer.is_fast:
        raise ValueError(
            f"Tokenizer for {model_name} is not a fast tokenizer. "
            f"word_ids() requires a fast tokenizer."
        )

    return tokenizer


def prepare_data(args, tokenizer):
    data = torch_load(args.data_path)

    required_keys = ["origin", "labels"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key in data: {key}")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive.")

        sliced = {}
        for key, value in data.items():
            if isinstance(value, list):
                sliced[key] = value[: args.max_samples]
            else:
                sliced[key] = value

        data = sliced
        print(f"Using max_samples={args.max_samples}")

    dataset = CompressionTokenDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_all_subtokens=not args.label_first_subtoken_only,
    )

    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    collator = make_collator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    metric_sums = {
        "accuracy": 0.0,
        "precision_keep": 0.0,
        "recall_keep": 0.0,
        "f1_keep": 0.0,
    }

    for batch in tqdm(val_loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        metrics = compute_token_metrics(outputs.logits, batch["labels"])

        total_loss += loss.item()
        total_batches += 1

        for key in metric_sums:
            metric_sums[key] += metrics[key]

    avg_loss = total_loss / max(total_batches, 1)

    avg_metrics = {
        key: value / max(total_batches, 1)
        for key, value in metric_sums.items()
    }

    avg_metrics["loss"] = avg_loss

    return avg_metrics


def save_training_config(args, save_path: str) -> None:
    config_path = Path(save_path) / "training_args.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if args.fp16 and device.type != "cuda":
        print("[WARN] --fp16 was set, but CUDA is not available. fp16 disabled.")
        args.fp16 = False

    tokenizer = load_tokenizer(args.model_name)

    train_loader, val_loader = prepare_data(args, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "DROP", 1: "KEEP"},
        label2id={"DROP": 0, "KEEP": 1},
    )

    model.to(device)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * args.num_epoch
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    save_training_config(args, str(save_path))

    best_f1 = -1.0
    global_step = 0

    for epoch in range(1, args.num_epoch + 1):
        model.train()

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epoch}")

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            running_loss += loss.item() * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    current_lr = scheduler.get_last_lr()[0]

                    progress.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )

                    running_loss = 0.0

        metrics = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch}] "
            f"val_loss={metrics['loss']:.6f} "
            f"acc={metrics['accuracy']:.6f} "
            f"precision_keep={metrics['precision_keep']:.6f} "
            f"recall_keep={metrics['recall_keep']:.6f} "
            f"f1_keep={metrics['f1_keep']:.6f}"
        )

        # Epoch checkpoints are disabled to save disk space.
        # Large XLM-RoBERTa checkpoints are ~2GB each.
        # We only save the best checkpoint by validation f1_keep.

        if metrics["f1_keep"] > best_f1:
            best_f1 = metrics["f1_keep"]

            best_path = save_path / "best"
            best_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            with open(best_path / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)

            print(f"Saved new best model to: {best_path}")

    print(f"Training finished. Best model saved to: {save_path / 'best'}")


if __name__ == "__main__":
    main()
