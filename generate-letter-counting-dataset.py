"""
Generates letter-counting chat samples and uploads them to the Hugging Face Hub.

Each sample is a conversation where the user asks how many times a letter appears
in a word, and the assistant responds using the explicit step-by-step procedure.

Usage:
    python generate_dataset.py --words words.txt --output my-hf-username/letter-count-dataset
"""

import random
import argparse
from datasets import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def spell_out(word: str) -> str:
    """Return the word as hyphen-separated characters, e.g. 's-t-r-a-w'."""
    return "-".join(list(word))


def build_assistant_response(word: str, letter: str) -> str:
    """
    Build the full step-by-step assistant response for counting `letter` in `word`.
    The count is strictly cumulative and never resets on a non-match.
    """
    lines = []

    # Step 1: spell out the word
    lines.append(f"Step 1 — Spell it out: {spell_out(word)}\n")

    # Step 2: check each position
    lines.append("Step 2 — Check each letter:")
    count = 0
    target = letter.lower()
    for i, char in enumerate(word, start=1):
        if char.lower() == target:
            count += 1
            lines.append(f"- Position {i}: {char} → MATCH (count = {count})")
        else:
            # Annotate the first few non-matches after a MATCH to reinforce
            # that the count does not reset. We annotate every non-match for clarity.
            lines.append(f"- Position {i}: {char} → not {letter} (count = {count})")

    # Final answer
    lines.append(
        f'\n**Answer: "{letter}" appears {count} time{"s" if count != 1 else ""} in "{word}".**'
    )

    return "\n".join(lines)


def build_sample(word: str, letter: str) -> dict:
    """
    Build a single chat sample as a list of role/content message dicts,
    ready to be used as a chat_template-compatible dataset entry.
    """
    user_message = f'How many times does the letter "{letter}" appear in "{word}"?'
    assistant_message = build_assistant_response(word, letter)

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------

def pick_letter_for_word(word: str, rng: random.Random) -> str:
    """
    Pick a letter to query for a given word.
    We bias toward letters that actually appear in the word (80% of the time)
    so the model also sees non-zero answers frequently. The remaining 20% picks
    a random a-z letter, which may or may not appear (teaches zero-count cases too).
    """
    if rng.random() < 0.8:
        return rng.choice(list(set(word.lower())))
    else:
        return rng.choice("abcdefghijklmnopqrstuvwxyz")


def generate_samples(
    words: list[str],
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """
    Generate `n_samples` chat samples by randomly pairing words with letters.
    Duplicates (same word + letter pair) are allowed but minimised by shuffling.
    """
    rng = random.Random(seed)
    
    # Cycle through words to spread coverage evenly
    word_pool = words * (n_samples // len(words) + 1)
    rng.shuffle(word_pool)

    return [
        build_sample(word, pick_letter_for_word(word, rng))
        for word in tqdm(word_pool[:n_samples], desc="Generating samples")
    ]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_words(path: str) -> list[str]:
    """Load one word per line from a text file, stripping whitespace and empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"No words found in {path}")
    
    # filter words that are too short or too long
    words = [word for word in words if len(word) >= 3 and len(word) <= 20]

    return words


def push_to_hub(samples: list[dict], repo_id: str, token: str | None = None) -> None:
    """Convert samples to a Hugging Face Dataset and push to the Hub."""
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(f"dataset/{repo_id}")
    print(f"\nDataset preview:\n{dataset}")
    print(f"\nSample entry:\n{samples[0]}")
    dataset.push_to_hub(repo_id, token=token)
    print(f"\nDataset pushed to: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate letter-counting chat dataset.")
    parser.add_argument("--words", required=True, help="Path to words.txt (one word per line)")
    parser.add_argument("--output", required=True, help="HF Hub repo id, e.g. username/dataset-name")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--token", default=None, help="HF API token (or set HF_TOKEN env var)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading words from {args.words}...")
    words = load_words(args.words)
    print(f"Loaded {len(words)} words.")

    print(f"Generating {args.n_samples} samples (seed={args.seed})...")
    samples = generate_samples(words, n_samples=args.n_samples, seed=args.seed)
    print(f"Generated {len(samples)} samples.")

    push_to_hub(samples, repo_id=args.output, token=args.token)


if __name__ == "__main__":
    main()
