import torch, random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from create_dataset import generate_dataset

# 1. Prepare your data
# If your data is a list of dicts, convert it to a Hugging Face Dataset
raw_data = generate_dataset(10000)
errors = []
for ex in raw_data:
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
random.shuffle(raw_data)
split = int(0.9 * len(raw_data))
train_data, val_data = raw_data[:split], raw_data[split:]

train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)
dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})

# 3. Push the entire dictionary at once
dataset_dict.push_to_hub("LastTransformer/add_numbers_grpo")

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
output_dir = "./smollm2-135m-sft-finetuned"

# 2. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # SmolLM2 needs a pad token defined

# 3. Define Formatting Function
# This wraps your prompt/completion into the ChatML format SmolLM2 expects
def format_instruction(sample):
    return {
        "prompt": [{"role": "user", "content": sample["prompt"]}],
        "completion": [
            {"role": "assistant", "content": sample["completion"]}
        ],
    }

# 4. Configure Training
sft_config = SFTConfig(
    output_dir=output_dir,
    max_length=512,
    completion_only_loss=True,
    packing=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    save_steps=100,
    logging_steps=10,
    push_to_hub=True,          # This enables automatic upload
    hub_model_id="LastTransformer/SmolLM2-135M-Custom-SFT", # Change this!
    report_to="none"
)

# Apply formatting to create a 'text' column
dataset = dataset.map(format_instruction)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model_id,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
    processing_class=tokenizer,
)

# 6. Train and Upload
trainer.train()
trainer.push_to_hub()
