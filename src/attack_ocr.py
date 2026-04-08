from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "prithivMLmods/Imgscope-OCR-2B-0527"
SOURCE_IMAGE_PATH = Path("data/images/worksheet_000000.png")
OCR_PROMPT = "Read all text in the image and output only the extracted text."
EPSILON = 32 / 255
ALPHA = 2 / 255
STEPS = 60
MAX_NEW_TOKENS = 256
RANDOM_START = False
RESULT_PREFIX = "imgscope_ocr_pgd"

MODEL_INPUT_SIZE = 448
TOKEN_TYPE_INPUT_KEYS = ("mm_token_type_ids", "token_type_ids")

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo root from the current working directory. "
        "Launch the script from this repository or one of its subdirectories."
    )

REPO_ROOT = find_repo_root()
RESULTS_DIR = REPO_ROOT / "results"
SOURCE_IMAGE_PATH = REPO_ROOT / SOURCE_IMAGE_PATH
OUTPUT_ADV_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_adv.png"
OUTPUT_REPORT_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_report.txt"

def load_image_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

def save_image_tensor(image_tensor: torch.Tensor, output_path: Path) -> None:
    image = image_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    array = (
        image.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array).save(output_path)

def pack_for_qwen(
    image_tensor: torch.Tensor,
    *,
    model_input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_tensor.shape[-2:]
    scale = min(model_input_size / height, model_input_size / width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))

    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_h = model_input_size - resized_h
    pad_w = model_input_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
    x = (x - mean) / std

    frames = x.unsqueeze(0)
    if frames.shape[0] % temporal_patch_size != 0:
        repeats = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
        frames = torch.cat([frames, frames[-1:].repeat(repeats, 1, 1, 1)], dim=0)

    channels = frames.shape[1]
    grid_t = frames.shape[0] // temporal_patch_size
    grid_h = frames.shape[2] // patch_size
    grid_w = frames.shape[3] // patch_size

    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    ).to(dtype=dtype)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=device, dtype=torch.long)
    return pixel_values, image_grid_thw

def build_text_model_inputs(
    prompt_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs = {
        "input_ids": prompt_inputs["input_ids"].to(device),
        "attention_mask": prompt_inputs["attention_mask"].to(device),
    }
    for token_type_key in TOKEN_TYPE_INPUT_KEYS:
        token_type_ids = prompt_inputs.get(token_type_key)
        if token_type_ids is not None:
            model_inputs[token_type_key] = token_type_ids.to(device)
    return model_inputs

def build_prompt_inputs(
    processor,
    device: torch.device,
    prompt: str,
) -> tuple[str, dict[str, torch.Tensor]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dummy_image = Image.new("RGB", (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), color="white")
    prompt_inputs = processor(
        text=[rendered_prompt],
        images=[dummy_image],
        return_tensors="pt",
    )
    return rendered_prompt, build_text_model_inputs(prompt_inputs, device)

def build_teacher_forced_inputs(
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    clean_text: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    target_ids = tokenizer(clean_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (target_ids.shape[1] == 0 or target_ids[0, -1].item() != eos_token_id):
        eos_tensor = torch.tensor([[eos_token_id]], device=device, dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    full_model_inputs = {
        "input_ids": torch.cat([prompt_model_inputs["input_ids"], target_ids], dim=1),
        "attention_mask": torch.cat(
            [prompt_model_inputs["attention_mask"], torch.ones_like(target_ids)],
            dim=1,
        ),
    }
    for token_type_key in TOKEN_TYPE_INPUT_KEYS:
        token_type_ids = prompt_model_inputs.get(token_type_key)
        if token_type_ids is not None:
            full_model_inputs[token_type_key] = torch.cat(
                [
                    token_type_ids,
                    torch.zeros(target_ids.shape, device=device, dtype=token_type_ids.dtype),
                ],
                dim=1,
            )

    labels = full_model_inputs["input_ids"].clone()
    labels[:, : prompt_model_inputs["input_ids"].shape[1]] = -100
    return full_model_inputs, labels

def build_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    pixel_values, image_grid_thw = pack_for_qwen(
        image_tensor,
        model_input_size=MODEL_INPUT_SIZE,
        mean=state["mean"].view(3, 1, 1),
        std=state["std"].view(3, 1, 1),
        patch_size=state["patch_size"],
        temporal_patch_size=state["temporal_patch_size"],
        merge_size=state["merge_size"],
        device=state["device"],
        dtype=state["dtype"],
    )
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

def generate_transcript(
    model,
    processor,
    prompt_model_inputs: dict[str, torch.Tensor],
    prompt_token_count: int,
    vision_inputs: dict[str, torch.Tensor],
) -> str:
    generated = model.generate(
        **prompt_model_inputs,
        **vision_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    new_tokens = generated[:, prompt_token_count:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

def transcription_loss(
    model,
    teacher_forced_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    vision_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        **teacher_forced_inputs,
        **vision_inputs,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss

def run_pgd(
    model,
    teacher_forced_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    state: dict,
    x_clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    delta = torch.zeros_like(x_clean, dtype=torch.float32)
    if RANDOM_START:
        delta.uniform_(-EPSILON, EPSILON)
        delta.clamp_(-EPSILON, EPSILON)
        delta.copy_(torch.clamp(x_clean + delta, 0.0, 1.0) - x_clean)
    delta.requires_grad_(True)

    saw_nonzero_grad = False
    last_loss = 0.0
    last_grad_inf = 0.0
    progress = tqdm(range(STEPS))
    for _ in progress:
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        vision_inputs = build_vision_inputs(state, x_adv.squeeze(0))
        loss = transcription_loss(model, teacher_forced_inputs, labels, vision_inputs)
        loss.backward()

        if delta.grad is None:
            raise RuntimeError("Expected PGD gradients on the perturbation tensor.")

        grad = delta.grad.detach()
        grad_inf = float(grad.abs().max().item())
        saw_nonzero_grad = saw_nonzero_grad or grad_inf > 0.0

        with torch.no_grad():
            delta.add_(ALPHA * grad.sign())
            delta.clamp_(-EPSILON, EPSILON)
            delta.copy_(torch.clamp(x_clean + delta, 0.0, 1.0) - x_clean)

        last_loss = float(loss.item())
        last_grad_inf = grad_inf
        progress.set_postfix(loss=f"{last_loss:.4f}", grad_inf=f"{last_grad_inf:.6f}")

    if not saw_nonzero_grad:
        raise RuntimeError("PGD never observed a non-zero image gradient.")

    x_final = torch.clamp(x_clean + delta.detach(), 0.0, 1.0)
    return x_final, delta.detach(), last_loss, last_grad_inf

def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

def build_report(
    clean_text: str,
    adv_text: str,
    *,
    final_loss: float,
    final_grad_inf: float,
    linf_delta: float,
) -> str:
    edit_distance = levenshtein_distance(clean_text, adv_text)
    normalized_edit_rate = edit_distance / max(1, len(clean_text))
    exact_match = clean_text == adv_text

    lines = [
        f"Model: {MODEL_NAME}",
        f"Prompt: {OCR_PROMPT}",
        f"Source image: {SOURCE_IMAGE_PATH.resolve()}",
        f"Epsilon: {EPSILON}",
        f"Alpha: {ALPHA}",
        f"Steps: {STEPS}",
        f"Random start: {RANDOM_START}",
        f"Final PGD loss: {final_loss:.6f}",
        f"Final gradient L_inf: {final_grad_inf:.6f}",
        f"Perturbation L_inf: {linf_delta:.6f}",
        f"Exact match: {exact_match}",
        f"Character edit distance: {edit_distance}",
        f"Normalized character edit rate: {normalized_edit_rate:.6f}",
        "",
        "Clean OCR text:",
        clean_text,
        "",
        "Adversarial OCR text:",
        adv_text,
        "",
        f"Adversarial image path: {OUTPUT_ADV_PATH.resolve()}",
    ]
    return "\n".join(lines)

def main() -> None:
    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Repo root: {REPO_ROOT}")
    print(f"Source image: {SOURCE_IMAGE_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    vision_config = model.config.vision_config
    state = {
        "device": device,
        "dtype": model_dtype,
        "patch_size": vision_config.patch_size,
        "temporal_patch_size": vision_config.temporal_patch_size,
        "merge_size": vision_config.spatial_merge_size,
        "mean": torch.tensor(processor.image_processor.image_mean, device=device, dtype=torch.float32),
        "std": torch.tensor(processor.image_processor.image_std, device=device, dtype=torch.float32),
    }

    _, prompt_model_inputs = build_prompt_inputs(processor, device, OCR_PROMPT)
    prompt_token_count = prompt_model_inputs["input_ids"].shape[1]

    x_clean = load_image_tensor(SOURCE_IMAGE_PATH, device)
    clean_vision_inputs = build_vision_inputs(state, x_clean.squeeze(0))
    with torch.no_grad():
        clean_text = generate_transcript(
            model,
            processor,
            prompt_model_inputs,
            prompt_token_count,
            clean_vision_inputs,
        )
    if not clean_text:
        raise RuntimeError("Clean OCR transcript was empty.")

    teacher_forced_inputs, labels = build_teacher_forced_inputs(
        processor.tokenizer,
        prompt_model_inputs,
        clean_text,
        device,
    )

    print("Running PGD...")
    x_adv, delta, final_loss, final_grad_inf = run_pgd(
        model,
        teacher_forced_inputs,
        labels,
        state,
        x_clean,
    )

    adv_vision_inputs = build_vision_inputs(state, x_adv.squeeze(0))
    with torch.no_grad():
        adv_text = generate_transcript(
            model,
            processor,
            prompt_model_inputs,
            prompt_token_count,
            adv_vision_inputs,
        )

    linf_delta = float(delta.abs().max().item())
    if linf_delta > EPSILON + 1e-6:
        raise RuntimeError(
            f"Perturbation exceeds the configured L_inf bound: {linf_delta:.6f} > {EPSILON:.6f}"
        )

    save_image_tensor(x_adv, OUTPUT_ADV_PATH)
    OUTPUT_REPORT_PATH.write_text(
        build_report(
            clean_text,
            adv_text,
            final_loss=final_loss,
            final_grad_inf=final_grad_inf,
            linf_delta=linf_delta,
        )
        + "\n"
    )

    print(f"Clean OCR: {clean_text}")
    print(f"Adversarial OCR: {adv_text}")
    print(f"Perturbation L_inf: {linf_delta:.6f}")
    print(f"Saved adversarial image to {OUTPUT_ADV_PATH.resolve()}")
    print(f"Saved report to {OUTPUT_REPORT_PATH.resolve()}")

if __name__ == "__main__":
    main()
