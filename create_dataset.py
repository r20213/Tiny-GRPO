import json
import random
from datasets import Dataset
import re

random.seed(42)

# ── Templates ────────────────────────────────────────────────────────────────
TEMPLATES = [
    # Animals
    "There are {a} chickens, {b} ducks and {c} cows in the barn.",
    "The farm has {a} horses, {b} sheep, and {c} goats.",
    "I saw {a} sparrows, {b} pigeons, and {c} crows on the wire.",
    "The zoo has {a} lions, {b} tigers, and {c} bears.",
    "There were {a} cats and {b} dogs at the shelter.",

    # Food
    "The basket contains {a} apples, {b} oranges, and {c} bananas.",
    "She bought {a} eggs, {b} loaves of bread, and {c} bottles of milk.",
    "The recipe calls for {a} cups of flour, {b} cups of sugar, and {c} eggs.",
    "He ate {a} cookies and drank {b} glasses of juice.",
    "The fridge has {a} apples, {b} pears, and {c} plums.",

    # People / Social
    "The class has {a} boys and {b} girls.",
    "There were {a} adults, {b} teenagers, and {c} children at the party.",
    "The team has {a} defenders, {b} midfielders, and {c} forwards.",
    "The office has {a} managers, {b} developers, and {c} designers.",
    "She invited {a} friends on Monday and {b} friends on Tuesday.",

    # Objects
    "The shelf holds {a} red books, {b} blue books, and {c} green books.",
    "The box contains {a} pens, {b} pencils, and {c} markers.",
    "There are {a} chairs, {b} tables, and {c} lamps in the room.",
    "He collected {a} stamps last week and {b} stamps this week.",
    "The store sold {a} shirts, {b} pants, and {c} jackets today.",

    # Nature / Travel
    "The garden has {a} roses, {b} tulips, and {c} daisies.",
    "On the hike we saw {a} deer, {b} rabbits, and {c} foxes.",
    "The river is {a} meters wide and {b} meters deep.",
    "She drove {a} miles on the highway and {b} miles on side roads.",
    "The park has {a} oak trees, {b} pine trees, and {c} maple trees.",

    # Two-number templates
    "He saved {a} dollars in January and {b} dollars in February.",
    "The library received {a} new fiction books and {b} new non-fiction books.",
    "There are {a} red cars and {b} blue cars in the lot.",
    "She ran {a} laps in the morning and {b} laps in the evening.",
    "The bag has {a} marbles and {b} coins.",

    # Four-number templates
    "The warehouse stores {a} boxes on floor one, {b} on floor two, {c} on floor three, and {d} on floor four.",
    "The school has {a} students in grade one, {b} in grade two, {c} in grade three, and {d} in grade four.",
    "She picked {a} strawberries, {b} blueberries, {c} raspberries, and {d} blackberries.",
    "The survey got {a} responses on Monday, {b} on Tuesday, {c} on Wednesday, and {d} on Thursday.",
]

INSTRUCTION = (
    "# Task : Add all the numbers in the paragraph below. "
    "Show your thought process inside <think></think> tags. "
    "The final answer should be inside <answer></answer> tags.\n"
    "# Paragraph : {paragraph}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_think(numbers: list[int]) -> str:
    """Build strict chain: a+b = s1 + c = s2 ..."""
    if len(numbers) == 1:
        return f"{numbers[0]}"
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


def extract_numbers(text: str) -> list[int]:
    return [int(m) for m in re.findall(r'\b\d+\b', text)]


def fill_template(template: str) -> tuple[str, list[int]]:
    """Fill a template with random numbers and return (paragraph, numbers)."""
    placeholders = re.findall(r'\{(\w)\}', template)
    unique_slots = sorted(set(placeholders), key=lambda x: placeholders.index(x))
    values = {}
    for slot in unique_slots:
        values[slot] = random.randint(1, 30)
    paragraph = template.format(**values)
    numbers = [values[p] for p in placeholders]  # preserve order of appearance
    return paragraph, numbers


def make_example(template: str) -> dict:
    paragraph, numbers = fill_template(template)
    think = build_think(numbers)
    answer = sum(numbers)
    prompt = INSTRUCTION.format(paragraph=paragraph)
    completion = f"<think>{think}</think><answer>{answer}</answer>"
    return {
        "prompt": prompt,
        "completion": completion,
        "numbers": numbers,
        "answer": answer,
    }


# ── Generate ──────────────────────────────────────────────────────────────────
def generate_dataset(n: int = 3000) -> list[dict]:
    examples = []
    seen_paragraphs = set()
    attempts = 0
    max_attempts = n * 10

    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        ex = make_example(template)
        # deduplicate on paragraph text
        if ex["prompt"] not in seen_paragraphs:
            seen_paragraphs.add(ex["prompt"])
            examples.append(ex)

    print(f"Generated {len(examples)} unique examples in {attempts} attempts.")
    return examples


if __name__ == "__main__":
    data = generate_dataset(10000)

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\n--- 3 random samples ---")
    for ex in random.sample(data, 3):
        print("PROMPT:")
        print(ex["prompt"])
        print("COMPLETION:")
        print(ex["completion"])
        print()

    # ── Verify think format ───────────────────────────────────────────────────
    errors = []
    for ex in data:
        nums = ex["numbers"]
        expected_think = build_think(nums)
        completion = ex["completion"]
        think_match = re.search(r'<think>(.*?)</think>', completion)
        answer_match = re.search(r'<answer>(\d+)</answer>', completion)
        if not think_match or not answer_match:
            errors.append(("missing tags", ex))
        elif think_match.group(1) != expected_think:
            errors.append(("think mismatch", ex))
        elif int(answer_match.group(1)) != ex["answer"]:
            errors.append(("answer mismatch", ex))
    print(f"Validation errors: {len(errors)} / {len(data)}")

    # ── Split & save ──────────────────────────────────────────────────────────
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    train_ds = Dataset.from_list(train_data)
    val_ds   = Dataset.from_list(val_data)

    train_ds.save_to_disk("/mnt/user-data/outputs/grpo_dataset/train")
    val_ds.save_to_disk("/mnt/user-data/outputs/grpo_dataset/val")

    # Also save as JSONL for easy inspection
    with open("/mnt/user-data/outputs/grpo_dataset/train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    with open("/mnt/user-data/outputs/grpo_dataset/val.jsonl", "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)}")
    print("Saved to /mnt/user-data/outputs/grpo_dataset/")
