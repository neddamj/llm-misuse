"""
1) Vision → Text: Extract math from image
   need a model that reads text/equations from an image. Some good Hugging Face models: Vision-Language Models (OCR + reasoning)
    (1) prithivMLmods/Imgscope-OCR-2B-0527  multimodal model fine-tuned on document and math problem reading tasks,
        outputs LaTeX/recognized text from images.
    (2) breezedeus/pix2text-mfr  formula recognition model that converts math images to LaTeX text.
    (3) OCR-focused VLMs (e.g., DeepSeek-OCR or other OCR models on Hugging Face) can extract printed/formula text.

2) Large Language Model for Math Reasoning
    After extracting the text, use a math-capable LLM to solve the problems.
    (1) Qwen2-Math  Instruction-tuned for mathematical reasoning
        Models: Qwen/Qwen2-Math-7B-Instruct, Qwen/Qwen2-Math-1.5B-Instruct, and larger.
        They’re trained to understand and solve math tasks, including multi-step reasoning.

"""


from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration #AutoModelForVision2Seq,
from PIL import Image
import torch

# Step 1: OCR/Formula Read
# ocr_model = "prithivMLmods/Imgscope-OCR-2B-0527"
# proc = AutoProcessor.from_pretrained(ocr_model, trust_remote_code=True)
# # ocr = AutoModelForVision2Seq.from_pretrained(ocr_model)
# ocr = AutoModelForCausalLM.from_pretrained(
#     ocr_model,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,  # or torch.float16 if you prefer
#     device_map="cuda"            # or remove and do ocr.to(device)
# )
# ocr.eval()

ocr_model = "prithivMLmods/Imgscope-OCR-2B-0527"

proc = AutoProcessor.from_pretrained(ocr_model, trust_remote_code=True)

ocr = Qwen2VLForConditionalGeneration.from_pretrained(
    ocr_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

ocr.eval()

# img = Image.open("math_page.jpg")
img = Image.open("../data/images/worksheet_000000.png").convert("RGB")
# ocr_inputs = proc(images=img, return_tensors="pt")
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Read all text in the image and output only the extracted text."},
    ],
}]

text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

ocr_inputs = proc(images=img, text=text, return_tensors="pt").to(device="cuda")


ocr_gen = ocr.generate(**ocr_inputs)
extracted = proc.batch_decode(ocr_gen, skip_special_tokens=True)[0]

print("Extracted Text:", extracted)

# Step 2: Math Solve
math_model = "Qwen/Qwen2-Math-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(math_model)
lm = AutoModelForCausalLM.from_pretrained(math_model, device_map="auto")

prompt = (
    "You are a math expert. Read the following problems and provide answers.\n"
    f"{extracted}\n"
    "Solve step by step."
)

inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
# out = lm.generate(**inputs, max_new_tokens=300, do_sample=False)

out = lm.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

solution = tokenizer.decode(out[0], skip_special_tokens=True)

print(solution)
