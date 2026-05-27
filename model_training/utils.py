import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForTokenClassification


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def lazy_load_spacy(model_name: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def split_string(text: Any, nlp=None) -> List[str]:
    """
    Same normalization logic as label_word.py:
    - skip whitespace and punctuation;
    - use lemma;
    - lowercase;
    - keep numbers.
    """
    if nlp is None:
        nlp = lazy_load_spacy()

    text = "" if text is None else str(text)
    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.strip()

        if not lemma or lemma == "-PRON-":
            lemma = token.text.strip()

        lemma = lemma.lower()

        if lemma:
            tokens.append(lemma)

    return tokens


class CompressionTokenDataset(Dataset):
    """
    Dataset for LLMLingua-2-style token classification.

    Input .pt is expected to contain at least:
        - origin: list[str]
        - labels: list[list[bool/int]]

    Preferably it also contains:
        - origin_tokens: list[list[str]]

    We tokenize origin_tokens with is_split_into_words=True, then expand word-level
    labels to tokenizer subword labels.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        tokenizer,
        max_length: int = 512,
        label_all_subtokens: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_subtokens = label_all_subtokens

        if "origin" not in data:
            raise KeyError("Input data must contain key: origin")

        if "labels" not in data:
            raise KeyError("Input data must contain key: labels")

        self.has_origin_tokens = "origin_tokens" in data

        self.nlp = None
        if not self.has_origin_tokens:
            self.nlp = lazy_load_spacy(spacy_model)

        self.valid_indices = []
        skipped = 0

        for idx in range(len(data["labels"])):
            labels = data["labels"][idx]

            if self.has_origin_tokens:
                words = data["origin_tokens"][idx]
            else:
                words = split_string(data["origin"][idx], self.nlp)

            if len(words) == 0:
                skipped += 1
                continue

            if len(words) != len(labels):
                skipped += 1
                continue

            self.valid_indices.append(idx)

        if len(self.valid_indices) == 0:
            raise ValueError("No valid samples found. Check origin_tokens and labels lengths.")

        if skipped > 0:
            print(f"[WARN] skipped invalid samples: {skipped}")

        print(f"Dataset samples: {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, item_idx: int) -> Dict[str, Any]:
        idx = self.valid_indices[item_idx]

        if self.has_origin_tokens:
            words = list(self.data["origin_tokens"][idx])
        else:
            words = split_string(self.data["origin"][idx], self.nlp)

        word_labels = [int(x) for x in self.data["labels"][idx]]

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        word_ids = encoded.word_ids()

        token_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)
            else:
                label = word_labels[word_id]

                if self.label_all_subtokens:
                    token_labels.append(label)
                else:
                    if word_id != previous_word_id:
                        token_labels.append(label)
                    else:
                        token_labels.append(-100)

            previous_word_id = word_id

        encoded["labels"] = token_labels
        return encoded


def make_collator(tokenizer):
    return DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1.")

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Dataset too small for the requested validation split.")

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


def compute_token_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes token-level metrics excluding labels == -100.
    Positive class is label 1 = KEEP token.
    """
    preds = torch.argmax(logits, dim=-1)

    mask = labels != -100

    if mask.sum().item() == 0:
        return {
            "accuracy": 0.0,
            "precision_keep": 0.0,
            "recall_keep": 0.0,
            "f1_keep": 0.0,
        }

    preds = preds[mask]
    labels = labels[mask]

    correct = (preds == labels).sum().item()
    total = labels.numel()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": correct / total,
        "precision_keep": precision,
        "recall_keep": recall,
        "f1_keep": f1,
    }
