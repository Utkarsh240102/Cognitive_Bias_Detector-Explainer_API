"""
Generate a synthetic cognitive bias dataset using Groq LLM.

Produces labeled examples for each of the 8 bias categories,
plus neutral (no-bias) and multi-label examples.
Output: data/dataset.csv
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ── Setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
OUTPUT_DIR = ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "dataset.csv"

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

# ── Prompts ─────────────────────────────────────────────────────

def _single_bias_prompt(bias: str, count: int) -> str:
    return (
        f"Generate exactly {count} unique, realistic sentences that clearly "
        f"demonstrate the cognitive bias called \"{bias}\".\n\n"
        f"Rules:\n"
        f"- Each sentence should be something a real person might say or write.\n"
        f"- Vary the topics (politics, work, relationships, health, society, etc.).\n"
        f"- Each sentence should be 1-2 sentences long.\n"
        f"- Do NOT include numbering, bullet points, or labels.\n"
        f"- Do NOT include explanations — only the biased statement itself.\n"
        f"- Output as a JSON array of strings, nothing else.\n\n"
        f"Example output format:\n"
        f'["sentence one", "sentence two", "sentence three"]'
    )


def _neutral_prompt(count: int) -> str:
    return (
        f"Generate exactly {count} unique, realistic sentences that contain "
        f"NO cognitive biases whatsoever. They should be neutral, factual, "
        f"or balanced statements.\n\n"
        f"Rules:\n"
        f"- Vary the topics (science, daily life, work, weather, etc.).\n"
        f"- Each sentence should be 1-2 sentences long.\n"
        f"- Do NOT include numbering, bullet points, or labels.\n"
        f"- Output as a JSON array of strings, nothing else.\n\n"
        f"Example output format:\n"
        f'["sentence one", "sentence two", "sentence three"]'
    )


def _multi_label_prompt(bias_pair: tuple[str, str], count: int) -> str:
    return (
        f"Generate exactly {count} unique, realistic sentences that demonstrate "
        f"BOTH of these cognitive biases simultaneously: "
        f"\"{bias_pair[0]}\" and \"{bias_pair[1]}\".\n\n"
        f"Rules:\n"
        f"- Each sentence should naturally exhibit both biases at the same time.\n"
        f"- Each sentence should be 1-2 sentences long.\n"
        f"- Do NOT include numbering, bullet points, or labels.\n"
        f"- Output as a JSON array of strings, nothing else.\n\n"
        f"Example output format:\n"
        f'["sentence one", "sentence two", "sentence three"]'
    )


# ── LLM Call ────────────────────────────────────────────────────

def _parse_sentences(raw: str) -> list[str]:
    """Try multiple strategies to extract sentences from LLM output."""
    # Strategy 1: Find and parse the JSON array directly
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            sentences = json.loads(match.group())
            if isinstance(sentences, list):
                return [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try to fix common JSON issues and re-parse
    if match:
        cleaned = match.group()
        # Remove trailing commas before ]
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        # Fix unescaped newlines inside strings
        cleaned = cleaned.replace("\n", " ")
        try:
            sentences = json.loads(cleaned)
            if isinstance(sentences, list):
                return [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Line-by-line extraction — grab quoted strings
    sentences = re.findall(r'"([^"]{15,})"', raw)
    if sentences:
        return [s.strip() for s in sentences if s.strip()]

    return []


def generate_sentences(client: Groq, prompt: str) -> list[str]:
    """Call Groq and parse the JSON array of sentences."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()

            sentences = _parse_sentences(raw)
            if sentences:
                return sentences

            print(f"  ⚠ Could not parse response (attempt {attempt + 1}), retrying...")
        except Exception as e:
            print(f"  ⚠ API error (attempt {attempt + 1}): {e}")
            time.sleep(5)

    return []


# ── Label Vector Helpers ────────────────────────────────────────

def make_label_vector(active_biases: list[str]) -> list[int]:
    """Return [0, 0, 1, 0, ...] with 1s for active biases."""
    return [1 if label in active_biases else 0 for label in BIAS_LABELS]


# ── Main ────────────────────────────────────────────────────────

def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_rows: list[dict] = []

    # ── 1. Single-bias examples (150 per bias) ──────────────────
    BATCH_SIZE = 50  # Generate in batches of 50 to avoid token limits
    PER_BIAS = 150

    for bias in BIAS_LABELS:
        print(f"\n📝 Generating {PER_BIAS} examples for: {bias}")
        collected = []

        for batch in range(PER_BIAS // BATCH_SIZE):
            print(f"  Batch {batch + 1}/{PER_BIAS // BATCH_SIZE}...")
            prompt = _single_bias_prompt(bias, BATCH_SIZE)
            sentences = generate_sentences(client, prompt)
            collected.extend(sentences)
            print(f"  Got {len(sentences)} sentences (total: {len(collected)})")
            time.sleep(2)  # Rate limiting

        label_vec = make_label_vector([bias])
        for sentence in collected:
            all_rows.append({"text": sentence, "labels": label_vec})

        print(f"  ✅ {bias}: {len(collected)} examples")

    # ── 2. Neutral examples (200) ───────────────────────────────
    print(f"\n📝 Generating 200 neutral (no-bias) examples")
    neutral_sentences = []
    for batch in range(4):
        print(f"  Batch {batch + 1}/4...")
        prompt = _neutral_prompt(50)
        sentences = generate_sentences(client, prompt)
        neutral_sentences.extend(sentences)
        print(f"  Got {len(sentences)} sentences (total: {len(neutral_sentences)})")
        time.sleep(2)

    label_vec = make_label_vector([])  # All zeros
    for sentence in neutral_sentences:
        all_rows.append({"text": sentence, "labels": label_vec})
    print(f"  ✅ Neutral: {len(neutral_sentences)} examples")

    # ── 3. Multi-label examples (common bias pairs) ─────────────
    MULTI_PAIRS = [
        ("Stereotyping", "Overgeneralization"),
        ("Stereotyping", "Hasty Generalization"),
        ("Overgeneralization", "Hasty Generalization"),
        ("False Dilemma", "Black-and-White Thinking"),
        ("Catastrophizing", "Emotional Reasoning"),
        ("Confirmation Bias", "Overgeneralization"),
    ]

    for pair in MULTI_PAIRS:
        print(f"\n📝 Generating 20 multi-label examples for: {pair[0]} + {pair[1]}")
        prompt = _multi_label_prompt(pair, 20)
        sentences = generate_sentences(client, prompt)

        label_vec = make_label_vector(list(pair))
        for sentence in sentences:
            all_rows.append({"text": sentence, "labels": label_vec})
        print(f"  ✅ {pair[0]} + {pair[1]}: {len(sentences)} examples")
        time.sleep(2)

    # ── 4. Write CSV ────────────────────────────────────────────
    print(f"\n💾 Writing {len(all_rows)} examples to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        header = ["text"] + BIAS_LABELS
        writer.writerow(header)
        # Rows
        for row in all_rows:
            writer.writerow([row["text"]] + row["labels"])

    print(f"\n🎉 Done! Generated {len(all_rows)} total examples.")
    print(f"   📁 Saved to: {OUTPUT_FILE}")

    # ── 5. Print stats ──────────────────────────────────────────
    print("\n📊 Label Distribution:")
    for i, label in enumerate(BIAS_LABELS):
        count = sum(1 for row in all_rows if row["labels"][i] == 1)
        print(f"   {label:30s} → {count} examples")
    neutral_count = sum(1 for row in all_rows if sum(row["labels"]) == 0)
    print(f"   {'Neutral (no bias)':30s} → {neutral_count} examples")


if __name__ == "__main__":
    main()
