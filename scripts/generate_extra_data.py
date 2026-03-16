"""
Generate additional targeted training examples to improve model accuracy.
Focuses on examples the model currently struggles with.
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
OUTPUT_FILE = ROOT / "data" / "extra_data.csv"

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


def _parse_sentences(raw: str) -> list[str]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            sentences = json.loads(match.group())
            if isinstance(sentences, list):
                return [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    if match:
        cleaned = match.group()
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        cleaned = cleaned.replace("\n", " ")
        try:
            sentences = json.loads(cleaned)
            if isinstance(sentences, list):
                return [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    sentences = re.findall(r'"([^"]{15,})"', raw)
    return [s.strip() for s in sentences if s.strip()]


def generate(client, prompt):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()
            sentences = _parse_sentences(raw)
            if sentences:
                return sentences
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(5)
    return []


def make_label_vector(active):
    return [1 if label in active else 0 for label in BIAS_LABELS]


# ── Targeted prompts for weak spots ────────────────────────────

TARGETED_PROMPTS = [
    # Stereotyping - gender
    {
        "prompt": "Generate 50 unique realistic sentences that stereotype people based on GENDER. Examples: 'Women are too emotional to lead', 'Men don't know how to express feelings'. Output JSON array only.",
        "labels": ["Stereotyping"],
    },
    # Stereotyping - race/ethnicity
    {
        "prompt": "Generate 50 unique realistic sentences that stereotype people based on RACE or ETHNICITY. Make them realistic things people actually say. Output JSON array only.",
        "labels": ["Stereotyping"],
    },
    # Stereotyping - age
    {
        "prompt": "Generate 50 unique realistic sentences that stereotype people based on AGE (young or old). Output JSON array only.",
        "labels": ["Stereotyping"],
    },
    # Stereotyping - profession/nationality
    {
        "prompt": "Generate 50 unique realistic sentences that stereotype people based on NATIONALITY or PROFESSION. Output JSON array only.",
        "labels": ["Stereotyping"],
    },
    # Overgeneralization
    {
        "prompt": "Generate 50 unique sentences demonstrating Overgeneralization - drawing broad universal conclusions from limited evidence. Use words like 'all', 'every', 'always', 'never', 'nobody'. Output JSON array only.",
        "labels": ["Overgeneralization"],
    },
    # Confirmation Bias
    {
        "prompt": "Generate 50 unique sentences demonstrating Confirmation Bias - selectively using evidence that supports pre-existing beliefs while ignoring contradictions. Output JSON array only.",
        "labels": ["Confirmation Bias"],
    },
    # Emotional Reasoning
    {
        "prompt": "Generate 50 unique sentences demonstrating Emotional Reasoning - treating feelings as proof that something is true. Use patterns like 'I feel X, therefore Y'. Output JSON array only.",
        "labels": ["Emotional Reasoning"],
    },
    # Catastrophizing
    {
        "prompt": "Generate 50 unique sentences demonstrating Catastrophizing - expecting the worst possible outcome from a normal situation. Output JSON array only.",
        "labels": ["Catastrophizing"],
    },
    # False Dilemma
    {
        "prompt": "Generate 50 unique sentences demonstrating False Dilemma - presenting only two options when more exist. Use 'either...or', 'if you don't...then'. Output JSON array only.",
        "labels": ["False Dilemma"],
    },
    # Black-and-White Thinking
    {
        "prompt": "Generate 50 unique sentences demonstrating Black-and-White Thinking - seeing situations in absolute extremes. Use 'completely', 'absolutely', 'totally', 'perfect/terrible'. Output JSON array only.",
        "labels": ["Black-and-White Thinking"],
    },
    # Hasty Generalization
    {
        "prompt": "Generate 50 unique sentences demonstrating Hasty Generalization - reaching broad conclusions from one or two incidents. Use patterns like 'I met one X who Y, so all X must Y'. Output JSON array only.",
        "labels": ["Hasty Generalization"],
    },
    # Multi-label: Stereotyping + Overgeneralization
    {
        "prompt": "Generate 30 unique sentences that demonstrate BOTH Stereotyping AND Overgeneralization simultaneously. Example: 'All women are bad at math'. Output JSON array only.",
        "labels": ["Stereotyping", "Overgeneralization"],
    },
    # More neutral examples
    {
        "prompt": "Generate 50 unique NEUTRAL sentences that contain NO cognitive bias. They should be balanced, factual statements about people, society, work, and daily life. Some should mention gender/race/age but in a fully neutral way. Output JSON array only.",
        "labels": [],
    },
]


def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    all_rows = []

    for item in TARGETED_PROMPTS:
        labels_str = " + ".join(item["labels"]) if item["labels"] else "Neutral"
        print(f"\n📝 Generating: {labels_str}")
        sentences = generate(client, item["prompt"])
        label_vec = make_label_vector(item["labels"])
        for s in sentences:
            all_rows.append({"text": s, "labels": label_vec})
        print(f"  ✅ Got {len(sentences)} sentences")
        time.sleep(2)

    # Write CSV
    print(f"\n💾 Writing {len(all_rows)} extra examples to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"] + BIAS_LABELS)
        for row in all_rows:
            writer.writerow([row["text"]] + row["labels"])

    print(f"🎉 Done! Generated {len(all_rows)} extra examples.")


if __name__ == "__main__":
    main()
