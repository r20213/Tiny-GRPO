import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from create_dataset import generate_dataset

# 1. Prepare your data
# If your data is a list of dicts, convert it to a Hugging Face Dataset
raw_data = generate_dataset(10000)[:10]
dataset = Dataset.from_list(raw_data)

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
output_dir = "./smollm2-135m-sft-finetuned"

# 2. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # SmolLM2 needs a pad token defined

# 3. Define Formatting Function
# This wraps your prompt/completion into the ChatML format SmolLM2 expects
def format_instruction(sample):
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]
    # This applies the model's specific chat template (e.g., <|im_start|>user...)
    return tokenizer.apply_chat_template(messages, tokenize=False)

# 4. Configure Training
sft_config = SFTConfig(
    output_dir=output_dir,
    max_length=512,
    dataset_text_field="text", # We will create this via a mapping
    packing=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    save_steps=100,
    logging_steps=10,
    push_to_hub=True,          # This enables automatic upload
    hub_model_id="LastTransformer/SmolLM2-135M-Custom-SFT", # Change this!
    report_to="none",
    train_on_completions=True,
)

# Apply formatting to create a 'text' column
dataset = dataset.map(lambda x: {"text": format_instruction(x)})

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

# 6. Train and Upload
trainer.train()
trainer.push_to_hub()
