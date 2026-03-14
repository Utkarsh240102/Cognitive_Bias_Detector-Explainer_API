"""
Evaluate and compare Zero-Shot BART vs. Fine-Tuned RoBERTa.

Tests both models on a set of validation sentences to compare:
- Accuracy (Exact matches)
- False Positive Rate
- Inference Speed
"""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizerFast

# ── Setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = ROOT / "data" / "val.csv"
TRAINED_MODEL = ROOT / "trained_model"

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

# ── Load Data ───────────────────────────────────────────────────

def load_val_data(max_samples: int = 100):
    """Load a subset of validation data for quick comparison."""
    df = pd.read_csv(VAL_CSV)
    df = df.sample(n=min(len(df), max_samples), random_state=42)
    texts = df["text"].tolist()
    labels = df.drop(columns=["text"]).values.tolist()
    return texts, labels


# ── Model Inferences ────────────────────────────────────────────

def run_zero_shot(texts: list[str]) -> tuple[list[list[int]], float]:
    """Run Meta's BART-large-mnli zero-shot model."""
    print("  Loading facebook/bart-large-mnli (zero-shot)...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    preds = []
    start_time = time.time()
    
    print("  Running inference...")
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(texts)}")
            
        result = classifier(text, candidate_labels=BIAS_LABELS)
        
        # Build binary prediction vector (threshold 0.5)
        pred_vec = [0] * len(BIAS_LABELS)
        for label, score in zip(result["labels"], result["scores"]):
            if score >= 0.5:
                idx = BIAS_LABELS.index(label)
                pred_vec[idx] = 1
        preds.append(pred_vec)
        
    duration = time.time() - start_time
    return preds, duration


def run_fine_tuned(texts: list[str]) -> tuple[list[list[int]], float]:
    """Run our custom fine-tuned RoBERTa model."""
    print("  Loading trained_model (RoBERTa multi-label)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = RobertaForSequenceClassification.from_pretrained(TRAINED_MODEL).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained(TRAINED_MODEL)
    model.eval()

    preds = []
    start_time = time.time()
    
    print("  Running inference...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(texts)}")
                
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            
            # Handle edge case where probs is a 0-d scalar (1 label, though we have 8)
            if probs.ndim == 0:
                probs = np.array([probs])
                
            pred_vec = (probs >= 0.5).astype(int).tolist()
            preds.append(pred_vec)

    duration = time.time() - start_time
    return preds, duration


# ── Evaluation ──────────────────────────────────────────────────

def evaluate_predictions(y_true: list[list[int]], y_pred: list[list[int]]) -> dict:
    """Calculate multi-label metrics."""
    # Convert to numpy arrays for easier math
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Calculate False Positive Rate (FPR)
    # FPR = FP / (FP + TN)
    fps = np.sum((y_pred_np == 1) & (y_true_np == 0))
    tns = np.sum((y_pred_np == 0) & (y_true_np == 0))
    fpr = fps / (fps + tns) if (fps + tns) > 0 else 0
    
    return {
        "F1 Score (Micro)": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Exact Match Acc.": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "False Pos. Rate": fpr,
    }


def print_comparison(metrics1: dict, dur1: float, metrics2: dict, dur2: float, n_samples: int):
    """Print head-to-head comparison table."""
    print("\n" + "="*70)
    print(f"  HEAD-TO-HEAD COMPARISON (on {n_samples} unseen validation sentences)")
    print("="*70)
    
    print(f"{'Metric':<25} | {'Zero-Shot (BART)':<20} | {'Fine-Tuned (RoBERTa)':<20}")
    print("-" * 70)
    
    for metric in metrics1.keys():
        val1 = metrics1[metric]
        val2 = metrics2[metric]
        
        # Format as percentage if it's a ratio <= 1.0
        s1 = f"{val1*100:.1f}%"
        s2 = f"{val2*100:.1f}%"
        
        # Highlight the winner
        if metric in ["False Pos. Rate"]:
            winner1 = val1 < val2
            winner2 = val2 < val1
        else:
            winner1 = val1 > val2
            winner2 = val2 > val1
            
        s1 = f"🏆 {s1}" if winner1 else f"   {s1}"
        s2 = f"🏆 {s2}" if winner2 else f"   {s2}"
        print(f"{metric:<25} | {s1:<20} | {s2:<20}")
        
    print("-" * 70)
    s1_speed = f"{dur1/n_samples*1000:.1f} ms/doc"
    s2_speed = f"{dur2/n_samples*1000:.1f} ms/doc"
    s1_speed = f"🏆 {s1_speed}" if dur1 < dur2 else f"   {s1_speed}"
    s2_speed = f"🏆 {s2_speed}" if dur2 < dur1 else f"   {s2_speed}"
    
    print(f"{'Inference Speed':<25} | {s1_speed:<20} | {s2_speed:<20}")
    print("="*70 + "\n")


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model Evaluation & Comparison")
    print("=" * 60)
    
    n_samples = 100  # Evaluate on 100 sentences to keep it quick
    print(f"\n📂 Loading {n_samples} sentences from validation set...")
    texts, y_true = load_val_data(max_samples=n_samples)
    
    print("\n🤖 Run 1: Fine-Tuned (RoBERTa-base)")
    ft_preds, ft_dur = run_fine_tuned(texts)
    ft_metrics = evaluate_predictions(y_true, ft_preds)
    
    print("\n🤖 Run 2: Zero-Shot (BART-large-mnli)")
    zs_preds, zs_dur = run_zero_shot(texts)
    zs_metrics = evaluate_predictions(y_true, zs_preds)
    
    # Print results
    print_comparison(zs_metrics, zs_dur, ft_metrics, ft_dur, n_samples)
    
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # Suppress HF warnings for clean output
    main()
