import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from PIL import Image

# --------------------------------------------------
# CUDA sanity check
# --------------------------------------------------
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

print("Using GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH = "../data/train.json"
IMAGE_DIR = "../data/images"

# --------------------------------------------------
# Load processor & model
# --------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

model.to(device)
model.train()

# Optional but helpful
model.gradient_checkpointing_enable()

# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def preprocess(example):
    image = Image.open(f"{IMAGE_DIR}/{example['image']}").convert("RGB")

    prompt = (
        "You are a math teacher. Solve the following problems step by step.\n"
        f"{example['question']}\n"
        "Answer:\n"
        f"{example['answer']}"
    )

    inputs = processor(
        images=image,
        text=prompt,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=1
)

# --------------------------------------------------
# Training arguments (single GPU)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./math_vlm_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",

    # SINGLE GPU SAFETY
    # no_cuda=False,
    # dataloader_pin_memory=True
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# --------------------------------------------------
# Train
# --------------------------------------------------
# trainer.train()

print("Training complete.")

#################### inference ##################

print("\n===== INFERENCE =====\n")
model.eval()

# --------------------------------------------------
# Load image
# --------------------------------------------------
image = Image.open("data/images/worksheet_000000.png").convert("RGB")

prompt = (
    "Solve the math problems shown in the image. "
    "Show steps and give the final answers."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # tells the template to insert the required image token(s)
            {"type": "text", "text": "Solve the math problems shown in the image. Show steps and give final answers."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------
# Prepare inputs
# --------------------------------------------------
inputs = processor(
    images=image,
    text=text,  # prompt,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# --------------------------------------------------
# Inference
# --------------------------------------------------
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

result = processor.decode(output_ids[0], skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====\n")
print(result)

