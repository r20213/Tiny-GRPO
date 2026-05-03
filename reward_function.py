"""
Reward functions for GRPO training on the number-addition task.

Each function has the signature expected by TRL's GRPOTrainer:
    fn(completions: list[str], **kwargs) -> list[float]

kwargs will contain any extra columns from the dataset. We use:
    - kwargs["numbers"]  : list[list[int]]  — the ground-truth operands
    - kwargs["answer"]   : list[int]        — the ground-truth sum
"""

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Small parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str | None:
    """Return the first inner content of <tag>…</tag>, or None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1) if m else None


def _normalize(text: str) -> str:
    """Collapse all whitespace to single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def _build_expected_think(numbers: list[int]) -> str:
    """
    Reconstruct the canonical think string for a number list.
    e.g. [1, 5, 6] → "1+5 = 6 + 6 = 12"
    Mirrors build_think() in create_dataset.py exactly.
    """
    if len(numbers) == 1:
        return str(numbers[0])
    parts = []
    running = numbers[0]
    for i, n in enumerate(numbers[1:], 1):
        new_sum = running + n
        if i == 1:
            parts.append(f"{running}+{n} = {new_sum}")
        else:
            parts.append(f"+ {n} = {new_sum}")
        running = new_sum
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Reward 1 — <think> tag presence
# ─────────────────────────────────────────────────────────────────────────────

def reward_think_tags(completions: list[str], **kwargs: Any) -> list[float]:
    """
    +0.1  if both <think> and </think> are present
    -0.1  otherwise
    """
    rewards = []
    for completion in completions:
        has_open  = "<think>"  in completion
        has_close = "</think>" in completion
        rewards.append(0.1 if (has_open and has_close) else -0.1)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Reward 2 — <think> content quality
# ─────────────────────────────────────────────────────────────────────────────

def reward_think_content(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Per-operand score (only the INPUT numbers, not intermediate sums):
        +0.02  for each expected operand found inside <think>
        -0.02  for each expected operand that is missing

    Exact-match bonus (after space-normalisation):
        +0.2   if think content matches the expected chain exactly
        -0.2   otherwise

    If <think> tags are absent the entire sub-reward is 0.0 (neutral) so
    this function doesn't double-penalise what reward_think_tags already caught.
    """
    numbers_batch: list[list[int]] = kwargs["numbers"]
    rewards = []

    for completion, numbers in zip(completions, numbers_batch):
        think_inner = _extract_tag(completion, "think")

        if think_inner is None:
            rewards.append(0.0)
            continue

        # ── Per-operand check ─────────────────────────────────────────────
        # Extract all integers that appear inside the think block
        found_nums = [int(m) for m in re.findall(r"\b\d+\b", think_inner)]

        # For each expected operand, check whether it appears in found_nums
        # (consume matches so duplicates in `numbers` are handled correctly)
        remaining = found_nums.copy()
        operand_score = 0.0
        for n in numbers:
            if n in remaining:
                remaining.remove(n)   # consume one occurrence
                operand_score += 0.02
            else:
                operand_score -= 0.02

        # ── Exact-match check ─────────────────────────────────────────────
        expected = _normalize(_build_expected_think(numbers))
        actual   = _normalize(think_inner)
        exact_score = 0.2 if (actual == expected) else -0.2

        rewards.append(operand_score + exact_score)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Reward 3 — <answer> tag presence + correctness (gated)
# ─────────────────────────────────────────────────────────────────────────────

def reward_answer(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Tag presence:
        +0.1   if both <answer> and </answer> are present
        -0.1   otherwise

    Correctness (only awarded when tags ARE present — no tags, no score):
        +1.0   answer inside tags matches ground truth
        -1.0   answer inside tags is wrong or non-numeric
    """
    answer_batch: list[int] = kwargs["answer"]
    rewards = []

    for completion, ground_truth in zip(completions, answer_batch):
        has_open  = "<answer>"  in completion
        has_close = "</answer>" in completion
        has_tags  = has_open and has_close

        tag_score = 0.1 if has_tags else -0.1

        has_think = "<think>" in completion and "</think>" in completion
        if not has_tags or not has_think:
            # Missing either tag block → no correctness signal
            # Prevents the model gaming +1.0 by skipping formatting entirely
            rewards.append(tag_score)
            continue

        inner = _extract_tag(completion, "answer")
        try:
            predicted = int(inner.strip())
            correctness_score = 1.0 if predicted == ground_truth else 0.01
        except (ValueError, AttributeError):
            correctness_score = -1.0

        rewards.append(tag_score + correctness_score)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Combined reward (optional — for logging / sanity checks)
# ─────────────────────────────────────────────────────────────────────────────

def reward_combined(completions: list[str], **kwargs: Any) -> list[float]:
    """Sum of all three reward components. Useful for eval logging."""
    r1 = reward_think_tags(completions, **kwargs)
    r2 = reward_think_content(completions, **kwargs)
    r3 = reward_answer(completions, **kwargs)
    return [a + b + c for a, b, c in zip(r1, r2, r3)]


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        {
            "label": "Perfect (3 numbers)",
            "completion": "<think>1+5 = 6 + 6 = 12</think><answer>12</answer>",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Perfect (2 numbers)",
            "completion": "<think>4+3 = 7</think><answer>7</answer>",
            "numbers": [4, 3],
            "answer": 7,
        },
        {
            "label": "Correct answer, messy think spacing",
            "completion": "<think>1 + 5  =  6 + 6 = 12</think><answer>12</answer>",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Wrong answer, correct think",
            "completion": "<think>1+5 = 6 + 6 = 12</think><answer>99</answer>",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Missing think tags, correct answer",
            "completion": "1+5=6+6=12<answer>12</answer>",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Missing think tags, wrong answer",
            "completion": "who knows<answer>99</answer>",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Missing answer tags",
            "completion": "<think>1+5 = 6 + 6 = 12</think>12",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "All tags missing",
            "completion": "The answer is 12.",
            "numbers": [1, 5, 6],
            "answer": 12,
        },
        {
            "label": "Perfect (4 numbers)",
            "completion": "<think>3+7 = 10 + 2 = 12 + 8 = 20</think><answer>20</answer>",
            "numbers": [3, 7, 2, 8],
            "answer": 20,
        },
    ]

    print(f"{'Label':<45} {'R1':>6} {'R2':>6} {'R3':>6} {'Total':>7}")
    print("-" * 75)
    for c in cases:
        completions = [c["completion"]]
        kw = {"numbers": [c["numbers"]], "answer": [c["answer"]]}
        r1 = reward_think_tags(completions, **kw)[0]
        r2 = reward_think_content(completions, **kw)[0]
        r3 = reward_answer(completions, **kw)[0]
        total = r1 + r2 + r3
        print(f"{c['label']:<45} {r1:>6.2f} {r2:>6.2f} {r3:>6.2f} {total:>7.2f}")
