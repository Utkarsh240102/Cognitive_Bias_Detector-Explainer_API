"""
Dataset quality check, deduplication, and train/val split.

Reads   : data/dataset.csv
Outputs : data/train.csv, data/val.csv
"""

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset.csv"
TRAIN_FILE = ROOT / "data" / "train.csv"
VAL_FILE = ROOT / "data" / "val.csv"

SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

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


def load_dataset(path: Path) -> tuple[list[str], list[list]]:
    """Load CSV and return (header, rows)."""
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def deduplicate(rows: list[list]) -> list[list]:
    """Remove duplicate and near-duplicate rows (by normalized text)."""
    seen: set[str] = set()
    unique = []

    for row in rows:
        # Normalize: lowercase, strip, collapse spaces
        text = row[0].lower().strip()
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(row)

    return unique


def check_label_validity(rows: list[list]) -> list[list]:
    """Remove rows with invalid labels (not 0 or 1)."""
    valid = []
    invalid_count = 0

    for row in rows:
        labels = row[1:]
        try:
            int_labels = [int(x) for x in labels]
            if all(x in (0, 1) for x in int_labels) and len(int_labels) == len(BIAS_LABELS):
                valid.append(row)
            else:
                invalid_count += 1
        except ValueError:
            invalid_count += 1

    if invalid_count:
        print(f"  ⚠ Removed {invalid_count} rows with invalid labels")

    return valid


def check_text_quality(rows: list[list]) -> list[list]:
    """Remove rows with very short or empty text."""
    valid = []
    removed = 0

    for row in rows:
        text = row[0].strip()
        if len(text) >= 15:  # At least 15 chars for a meaningful sentence
            valid.append(row)
        else:
            removed += 1

    if removed:
        print(f"  ⚠ Removed {removed} rows with text < 15 chars")

    return valid


def print_stats(rows: list[list]):
    """Print label distribution and dataset statistics."""
    total = len(rows)
    print(f"\n📊 Dataset Statistics ({total} examples):")
    print(f"{'─' * 55}")

    for i, label in enumerate(BIAS_LABELS):
        count = sum(1 for row in rows if int(row[i + 1]) == 1)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:30s}  {count:4d} ({pct:5.1f}%)  {bar}")

    neutral = sum(1 for row in rows if all(int(x) == 0 for x in row[1:]))
    print(f"  {'Neutral (no bias)':30s}  {neutral:4d} ({neutral / total * 100:5.1f}%)")

    # Multi-label stats
    label_counts = Counter()
    for row in rows:
        n_labels = sum(int(x) for x in row[1:])
        label_counts[n_labels] += 1

    print(f"\n  Label count distribution:")
    for n in sorted(label_counts):
        desc = "neutral" if n == 0 else f"{n} bias{'es' if n > 1 else ''}"
        print(f"    {desc:20s} → {label_counts[n]} examples")


def split_dataset(rows: list[list]) -> tuple[list[list], list[list]]:
    """Stratified-ish random split into train and val."""
    random.seed(SEED)
    shuffled = rows.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * SPLIT_RATIO)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    return train, val


def save_csv(path: Path, header: list[str], rows: list[list]):
    """Write rows to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    print("🔍 Loading dataset...")
    header, rows = load_dataset(DATASET)
    print(f"  Loaded {len(rows)} rows")

    # Step 1: Deduplicate
    print("\n🧹 Deduplicating...")
    before = len(rows)
    rows = deduplicate(rows)
    removed = before - len(rows)
    print(f"  Removed {removed} duplicates → {len(rows)} unique rows")

    # Step 2: Validate labels
    print("\n✅ Checking label validity...")
    rows = check_label_validity(rows)

    # Step 3: Check text quality
    print("\n📏 Checking text quality...")
    rows = check_text_quality(rows)

    # Step 4: Print stats
    print_stats(rows)

    # Step 5: Split
    train, val = split_dataset(rows)
    print(f"\n✂️  Split: {len(train)} train / {len(val)} val")

    # Step 6: Save
    save_csv(TRAIN_FILE, header, train)
    save_csv(VAL_FILE, header, val)
    print(f"  💾 Saved: {TRAIN_FILE}")
    print(f"  💾 Saved: {VAL_FILE}")

    print(f"\n🎉 Done! Ready for training.")


if __name__ == "__main__":
    main()
