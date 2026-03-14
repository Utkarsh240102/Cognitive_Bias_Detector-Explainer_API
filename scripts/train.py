"""
Fine-tune RoBERTa-base for multi-label cognitive bias classification.

Reads   : data/train.csv, data/val.csv
Outputs : trained_model/  (model + tokenizer + config)

Optimised for RTX 2050 (4 GB VRAM):
  - batch_size  = 4
  - fp16        = True
  - max_length  = 128
"""

import csv
import numpy as np
import torch
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

# ── Paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
VAL_CSV = ROOT / "data" / "val.csv"
OUTPUT_DIR = ROOT / "trained_model"
RESULTS_DIR = ROOT / "results"

# ── Hyper-parameters ────────────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128          # Token length (128 is plenty for 1-2 sentence inputs)
BATCH_SIZE = 4            # Fits in 4 GB VRAM
LEARNING_RATE = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01

BIAS_LABELS = [
    "Confirmation Bias",
    "Overgeneralization",
    "Stereotyping",
    "Emotional Reasoning",
    "False Dilemma",
    "Catastrophizing",
    "Hasty Generalization",
    "Black-and-White Thinking",
]
NUM_LABELS = len(BIAS_LABELS)


# ── Dataset ─────────────────────────────────────────────────────

class BiasDataset(torch.utils.data.Dataset):
    """PyTorch dataset for multi-label bias classification."""

    def __init__(self, texts: list[str], labels: list[list[int]], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_csv(path: Path) -> tuple[list[str], list[list[int]]]:
    """Load CSV and return (texts, label_vectors)."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            texts.append(row[0])
            labels.append([int(x) for x in row[1:]])
    return texts, labels


# ── Metrics ─────────────────────────────────────────────────────

def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    """Compute multi-label classification metrics."""
    logits = pred.predictions
    labels = pred.label_ids

    # Apply sigmoid + threshold 0.5
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_samples": f1_score(labels, preds, average="samples", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
        "exact_match": accuracy_score(labels, preds),
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RoBERTa Fine-Tuning for Cognitive Bias Detection")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\n🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n⚠️  No GPU detected — training on CPU (will be slower)")

    # Load data
    print("\n📂 Loading datasets...")
    train_texts, train_labels = load_csv(TRAIN_CSV)
    val_texts, val_labels = load_csv(VAL_CSV)
    print(f"  Train: {len(train_texts)} examples")
    print(f"  Val:   {len(val_texts)} examples")

    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    # Create datasets
    print("📦 Tokenizing datasets...")
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)

    # Load model
    print(f"\n🧠 Loading model: {MODEL_NAME} ({NUM_LABELS} labels, multi-label)")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=25,
        logging_dir=str(RESULTS_DIR / "logs"),
        report_to="none",  # No wandb/tensorboard
        save_total_limit=2,  # Keep only 2 best checkpoints to save disk
    )

    print(f"\n⚙️  Training config:")
    print(f"    Epochs:         {EPOCHS}")
    print(f"    Batch size:     {BATCH_SIZE}")
    print(f"    Learning rate:  {LEARNING_RATE}")
    print(f"    FP16:           {training_args.fp16}")
    print(f"    Max length:     {MAX_LENGTH}")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\n🚀 Starting training...")
    print("─" * 60)
    trainer.train()
    print("─" * 60)

    # Final evaluation
    print("\n📊 Final evaluation on validation set:")
    metrics = trainer.evaluate()
    for key, value in sorted(metrics.items()):
        if key.startswith("eval_"):
            name = key.replace("eval_", "")
            print(f"    {name:20s} = {value:.4f}")

    # Save best model + tokenizer
    print(f"\n💾 Saving best model to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save label mapping
    label_map = {i: label for i, label in enumerate(BIAS_LABELS)}
    import json
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n🎉 Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Label map saved to: {OUTPUT_DIR / 'label_map.json'}")


if __name__ == "__main__":
    main()
