from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from attacks.common import find_repo_root, load_image_tensor, project_delta, save_image_tensor
from attacks.prompting import build_chat_prompt_inputs, build_teacher_forced_batch, generate_greedy_text
from attacks.vision import build_qwen_vision_inputs

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

REPO_ROOT = find_repo_root(Path(__file__).resolve())
RESULTS_DIR = REPO_ROOT / "results"
SOURCE_IMAGE_PATH = REPO_ROOT / SOURCE_IMAGE_PATH
OUTPUT_ADV_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_adv.png"
OUTPUT_REPORT_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_report.txt"

def transcription_loss(
    model,
    teacher_forced_batch: dict[str, torch.Tensor],
    vision_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        **teacher_forced_batch["model_inputs"],
        **vision_inputs,
        labels=teacher_forced_batch["labels"],
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss

def run_pgd(
    model,
    teacher_forced_batch: dict[str, torch.Tensor],
    state: dict,
    x_clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    delta = torch.zeros_like(x_clean, dtype=torch.float32)
    if RANDOM_START:
        delta.uniform_(-EPSILON, EPSILON)
        project_delta(delta, x_clean, EPSILON)
    delta.requires_grad_(True)

    saw_nonzero_grad = False
    last_loss = 0.0
    last_grad_inf = 0.0
    progress = tqdm(range(STEPS))
    for _ in progress:
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        vision_inputs = build_qwen_vision_inputs(state, x_adv.squeeze(0))
        loss = transcription_loss(model, teacher_forced_batch, vision_inputs)
        loss.backward()

        if delta.grad is None:
            raise RuntimeError("Expected PGD gradients on the perturbation tensor.")

        grad = delta.grad.detach()
        grad_inf = float(grad.abs().max().item())
        saw_nonzero_grad = saw_nonzero_grad or grad_inf > 0.0

        with torch.no_grad():
            delta.add_(ALPHA * grad.sign())
            project_delta(delta, x_clean, EPSILON)

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
        "model_input_size": MODEL_INPUT_SIZE,
        "patch_size": vision_config.patch_size,
        "temporal_patch_size": vision_config.temporal_patch_size,
        "merge_size": vision_config.spatial_merge_size,
        "mean": torch.tensor(processor.image_processor.image_mean, device=device, dtype=torch.float32),
        "std": torch.tensor(processor.image_processor.image_std, device=device, dtype=torch.float32),
    }

    _, prompt_model_inputs = build_chat_prompt_inputs(
        processor,
        device,
        OCR_PROMPT,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
    )
    prompt_token_count = prompt_model_inputs["input_ids"].shape[1]

    x_clean = load_image_tensor(SOURCE_IMAGE_PATH, device)
    clean_vision_inputs = build_qwen_vision_inputs(state, x_clean.squeeze(0))
    with torch.no_grad():
        clean_text = generate_greedy_text(
            model,
            processor,
            prompt_model_inputs,
            prompt_token_count,
            clean_vision_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    if not clean_text:
        raise RuntimeError("Clean OCR transcript was empty.")

    teacher_forced_batch = build_teacher_forced_batch(
        processor.tokenizer,
        prompt_model_inputs,
        clean_text,
        device,
    )

    print("Running PGD...")
    x_adv, delta, final_loss, final_grad_inf = run_pgd(
        model,
        teacher_forced_batch,
        state,
        x_clean,
    )

    adv_vision_inputs = build_qwen_vision_inputs(state, x_adv.squeeze(0))
    with torch.no_grad():
        adv_text = generate_greedy_text(
            model,
            processor,
            prompt_model_inputs,
            prompt_token_count,
            adv_vision_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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
