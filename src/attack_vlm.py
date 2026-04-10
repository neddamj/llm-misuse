from pathlib import Path

import torch
import torch.multiprocessing as mp
from torchvision import transforms
from tqdm import tqdm
from attacks.common import (
    canonicalize_cuda_device,
    find_repo_root,
    load_image_tensor,
    project_delta,
    summarize_loss_values,
)
from attacks.vision import sample_camera_transform
from attacks.workers import (
    attack_workers,
    evaluate_workers,
    set_workers_untargeted_references,
    shutdown_workers,
    start_workers,
)


REPO_ROOT = find_repo_root()
RESULTS_DIR = REPO_ROOT / "results"

MODEL_SPECS = [
    {
        "key": "granite_vision_3_2",
        "model_name": "ibm-granite/granite-vision-3.2-2b",
        "model_family": "auto",
        "device": "cuda:0",
    },
    {
        "key": "gemma3_4b_it",
        "model_name": "google/gemma-3-4b-it",
        "model_family": "auto",
        "device": "cuda:1",
    },
    {
        "key": "smolvlm_2b_instruct",
        "model_name": "HuggingFaceTB/SmolVLM-Instruct",
        "model_family": "auto",
        "device": "cuda:2",
    },
]


SOURCE_IMAGE_PATH = REPO_ROOT / "data" / "images" / "worksheet_000002.png"
USER_PROMPT = "Solve the math problems shown in the image. Show steps and give the final answers."
ATTACK_MODE = "untargeted" # Options: "targeted", "untargeted"
TARGET_TEXTS: list[str] = [
    "I cannot help with that",
    "No",
    "I'm sorry, I cant answer that.",
]
TARGET_LOSS_MODE = "standard_ce" # Options: "standard_ce", "multi_reference"
CROSS_MODEL_OPTIMIZATION_MODE = "mean_ce" # Options: "mean_ce", "softminimax"

EPSILON = 64 / 255
ALPHA = 4 / 1000
STEPS = 1500
ATTACK_IMAGE_SIZE = (400, 400)
MODEL_INPUT_SIZE = 448
MAX_NEW_TOKENS = 128
CROSS_MODEL_SOFTMINIMAX_TEMPERATURE = 1.0
USE_EOT = False
EOT_TRAIN_SAMPLES = 4
EOT_EVAL_SAMPLES = 1
EOT_ROTATION_DEGREES = 0 #5
EOT_PERSPECTIVE_DISTORTION = 0 #0.2
EOT_CROP_SCALE = (1.0, 1.0)#(0.8, 1.0)
EOT_CROP_RATIO = (1.0, 1.0)#(0.9, 1.1)
EOT_COLOR_JITTER_BRIGHTNESS = 0.1
EOT_COLOR_JITTER_CONTRAST = 0.1
EOT_COLOR_JITTER_SATURATION = 0.1
EOT_GAUSSIAN_NOISE_STD = 0.02

RESULT_PREFIX = "granite_gemma3_smolvlm_textgen_multi_gpu"
OUTPUT_ADV_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_adv.png"
OUTPUT_NOISE_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_noise.png"
OUTPUT_REPORT_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_generations.txt"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SUPPORTED_ATTACK_MODES = {"targeted", "untargeted"}
SUPPORTED_TARGET_LOSS_MODES = {"multi_reference", "standard_ce"}
SUPPORTED_CROSS_MODEL_OPTIMIZATION_MODES = {"mean_ce", "softminimax"}


def save_noise_visualization(delta: torch.Tensor, output_path: Path) -> None:
    noise = torch.clamp(delta.squeeze(0).cpu() * 10 + 0.5, 0.0, 1.0)
    transforms.ToPILImage()(noise).save(output_path)


def validate_config() -> None:
    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE_PATH}")

    if ATTACK_MODE not in SUPPORTED_ATTACK_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_ATTACK_MODES))
        raise ValueError(
            f"ATTACK_MODE must be one of: {supported_modes}. "
            f"Got {ATTACK_MODE!r}."
        )

    if ATTACK_MODE == "targeted" and TARGET_LOSS_MODE not in SUPPORTED_TARGET_LOSS_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_TARGET_LOSS_MODES))
        raise ValueError(
            f"TARGET_LOSS_MODE must be one of: {supported_modes}. "
            f"Got {TARGET_LOSS_MODE!r}."
        )

    if CROSS_MODEL_OPTIMIZATION_MODE not in SUPPORTED_CROSS_MODEL_OPTIMIZATION_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_CROSS_MODEL_OPTIMIZATION_MODES))
        raise ValueError(
            f"CROSS_MODEL_OPTIMIZATION_MODE must be one of: {supported_modes}. "
            f"Got {CROSS_MODEL_OPTIMIZATION_MODE!r}."
        )

    if ATTACK_MODE == "targeted" and (
        not TARGET_TEXTS
        or any(not isinstance(target_text, str) or not target_text for target_text in TARGET_TEXTS)
    ):
        raise ValueError("TARGET_TEXTS must be a non-empty list of non-empty strings.")

    if (
        CROSS_MODEL_OPTIMIZATION_MODE == "softminimax"
        and CROSS_MODEL_SOFTMINIMAX_TEMPERATURE <= 0
    ):
        raise ValueError("CROSS_MODEL_SOFTMINIMAX_TEMPERATURE must be positive.")

    if USE_EOT:
        if EOT_TRAIN_SAMPLES <= 0:
            raise ValueError("EOT_TRAIN_SAMPLES must be positive when USE_EOT is enabled.")
        if EOT_EVAL_SAMPLES <= 0:
            raise ValueError("EOT_EVAL_SAMPLES must be positive when USE_EOT is enabled.")

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    visible_gpu_count = torch.cuda.device_count()
    required_gpu_count = len(MODEL_SPECS)
    if visible_gpu_count < required_gpu_count:
        raise RuntimeError(
            f"This script requires at least {required_gpu_count} visible GPUs, "
            f"but only found {visible_gpu_count}."
        )

    devices = [canonicalize_cuda_device(model_spec["device"]) for model_spec in MODEL_SPECS]
    if len(set(devices)) != len(devices):
        raise RuntimeError("Each model must be assigned to a distinct CUDA device.")

    for device_name in devices:
        device_index = torch.device(device_name).index
        if device_index is None or device_index >= visible_gpu_count:
            raise RuntimeError(
                f"Configured device {device_name} is not visible. "
                f"Visible GPU count: {visible_gpu_count}."
            )
def get_configured_target_texts() -> list[str]:
    if ATTACK_MODE != "targeted":
        return []
    if TARGET_LOSS_MODE == "standard_ce":
        return [TARGET_TEXTS[0]]
    return TARGET_TEXTS


def compute_cross_model_aggregation(
    ordered_keys: list[str],
    metric_losses_by_key: dict[str, float],
    optimization_losses_by_key: dict[str, float],
    gradients: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, str, float, float]:
    if not ordered_keys or not gradients:
        raise RuntimeError("Expected at least one worker result for cross-model aggregation.")

    metric_losses = torch.tensor(
        [metric_losses_by_key[key] for key in ordered_keys],
        device=device,
        dtype=dtype,
    )
    optimization_losses = torch.tensor(
        [optimization_losses_by_key[key] for key in ordered_keys],
        device=device,
        dtype=dtype,
    )
    stacked_grads = torch.stack(gradients, dim=0).to(device=device, dtype=dtype)
    worst_index = int(torch.argmax(optimization_losses).item())

    if CROSS_MODEL_OPTIMIZATION_MODE == "softminimax":
        temperature = torch.tensor(
            CROSS_MODEL_SOFTMINIMAX_TEMPERATURE,
            device=device,
            dtype=dtype,
        )
        weights = torch.softmax(optimization_losses / temperature, dim=0)
        weight_shape = (len(ordered_keys),) + (1,) * (stacked_grads.ndim - 1)
        aggregated_grad = (stacked_grads * weights.view(weight_shape)).sum(dim=0)
        aggregate_loss = float(
            (
                temperature
                * (
                    torch.logsumexp(optimization_losses / temperature, dim=0)
                    - optimization_losses.new_tensor(len(ordered_keys)).log()
                )
            ).item()
        )
        return (
            aggregated_grad,
            ordered_keys[worst_index],
            float(metric_losses[worst_index].item()),
            aggregate_loss,
        )

    if CROSS_MODEL_OPTIMIZATION_MODE == "mean_ce":
        aggregated_grad = stacked_grads.mean(dim=0)
        aggregate_loss = float(optimization_losses.mean().item())
        return (
            aggregated_grad,
            ordered_keys[worst_index],
            float(metric_losses[worst_index].item()),
            aggregate_loss,
        )

    raise ValueError(
        f"Unsupported CROSS_MODEL_OPTIMIZATION_MODE: {CROSS_MODEL_OPTIMIZATION_MODE!r}"
    )


def evaluate_workers_eot(
    workers: list[dict],
    image_cpu: torch.Tensor,
    *,
    num_samples: int,
) -> dict:
    loss_sums = {model_spec["key"]: 0.0 for model_spec in MODEL_SPECS}
    for _ in range(num_samples):
        with torch.no_grad():
            transformed_image = sample_camera_transform(
                image_cpu.squeeze(0),
                rotation_degrees=EOT_ROTATION_DEGREES,
                perspective_distortion=EOT_PERSPECTIVE_DISTORTION,
                crop_scale=EOT_CROP_SCALE,
                crop_ratio=EOT_CROP_RATIO,
                color_jitter_brightness=EOT_COLOR_JITTER_BRIGHTNESS,
                color_jitter_contrast=EOT_COLOR_JITTER_CONTRAST,
                color_jitter_saturation=EOT_COLOR_JITTER_SATURATION,
                gaussian_noise_std=EOT_GAUSSIAN_NOISE_STD,
            ).unsqueeze(0)
        sample_results = evaluate_workers(workers, transformed_image)
        for key, result in sample_results.items():
            loss_sums[key] += result["loss"]

    per_model_mean_losses = {key: loss_sums[key] / num_samples for key in loss_sums}
    worst_key, worst_loss, mean_loss = summarize_loss_values(
        per_model_mean_losses,
        higher_is_worse=ATTACK_MODE == "targeted",
    )
    return {
        "num_samples": num_samples,
        "per_model_mean_losses": per_model_mean_losses,
        "worst_key": worst_key,
        "worst_loss": worst_loss,
        "mean_loss": mean_loss,
    }
def build_progress_postfix(
    losses_by_key: dict[str, float],
    worst_key: str,
    worst_loss: float,
    aggregate_loss: float,
    mean_loss: float,
    *,
    prefix: str = "",
) -> dict[str, str]:
    aggregate_key = "aggregate_loss" if ATTACK_MODE == "targeted" else "aggregate_objective"
    postfix = {f"{key}_{prefix}loss": f"{value:.4f}" for key, value in losses_by_key.items()}
    postfix.update(
        {
            f"{prefix}worst_model": worst_key,
            f"{prefix}worst_loss": f"{worst_loss:.4f}",
            f"{prefix}{aggregate_key}": f"{aggregate_loss:.4f}",
            f"{prefix}mean_loss": f"{mean_loss:.4f}",
        }
    )
    return postfix


def get_metric_loss_label() -> str:
    if ATTACK_MODE == "untargeted":
        return "untargeted reference loss"
    return "target loss"


def run_attack(workers: list[dict], x_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.zeros_like(x_clean, requires_grad=True)
    optimizer = torch.optim.AdamW([delta], lr=ALPHA, weight_decay=0.0)
    progress = tqdm(range(STEPS))

    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        if not USE_EOT:
            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
            ordered_keys, step_results, optimization_losses, gradients = attack_workers(workers, x_adv)
            aggregated_grad, worst_key, worst_loss, aggregate_loss = compute_cross_model_aggregation(
                ordered_keys,
                step_results,
                optimization_losses,
                gradients,
                device=delta.device,
                dtype=delta.dtype,
            )
            delta.grad = aggregated_grad.detach()
            optimizer.step()
            project_delta(delta, x_clean, EPSILON)

            _, _, mean_loss = summarize_loss_values(
                step_results,
                higher_is_worse=ATTACK_MODE == "targeted",
            )
            progress.set_postfix(
                build_progress_postfix(
                    step_results,
                    worst_key,
                    worst_loss,
                    aggregate_loss,
                    mean_loss,
                )
            )
            continue

        eot_loss_sums = {model_spec["key"]: 0.0 for model_spec in MODEL_SPECS}
        eot_aggregate_loss = 0.0
        for _ in range(EOT_TRAIN_SAMPLES):
            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
            transformed_image = sample_camera_transform(
                x_adv.squeeze(0),
                rotation_degrees=EOT_ROTATION_DEGREES,
                perspective_distortion=EOT_PERSPECTIVE_DISTORTION,
                crop_scale=EOT_CROP_SCALE,
                crop_ratio=EOT_CROP_RATIO,
                color_jitter_brightness=EOT_COLOR_JITTER_BRIGHTNESS,
                color_jitter_contrast=EOT_COLOR_JITTER_CONTRAST,
                color_jitter_saturation=EOT_COLOR_JITTER_SATURATION,
                gaussian_noise_std=EOT_GAUSSIAN_NOISE_STD,
            ).unsqueeze(0)
            ordered_keys, step_results, optimization_losses, gradients = attack_workers(workers, transformed_image)
            aggregated_grad, _, _, sample_aggregate_loss = compute_cross_model_aggregation(
                ordered_keys,
                step_results,
                optimization_losses,
                gradients,
                device=delta.device,
                dtype=delta.dtype,
            )
            transformed_image.backward(aggregated_grad)
            eot_aggregate_loss += sample_aggregate_loss
            for key, loss in step_results.items():
                eot_loss_sums[key] += loss

        if delta.grad is None:
            raise RuntimeError("Expected EoT gradients on the perturbation tensor.")

        delta.grad.div_(EOT_TRAIN_SAMPLES)
        optimizer.step()
        project_delta(delta, x_clean, EPSILON)

        eot_step_results = {key: eot_loss_sums[key] / EOT_TRAIN_SAMPLES for key in eot_loss_sums}
        eot_worst_key, eot_worst_loss, eot_mean_loss = summarize_loss_values(
            eot_step_results,
            higher_is_worse=ATTACK_MODE == "targeted",
        )
        eot_aggregate_loss /= EOT_TRAIN_SAMPLES
        progress.set_postfix(
            build_progress_postfix(
                eot_step_results,
                eot_worst_key,
                eot_worst_loss,
                eot_aggregate_loss,
                eot_mean_loss,
                prefix="eot_",
            )
        )

    return torch.clamp(x_clean + delta, 0.0, 1.0).detach(), delta.detach()


def build_target_config_lines(
    configured_target_texts: list[str],
    *,
    target_first: bool,
) -> list[str]:
    target_lines: list[str]
    if ATTACK_MODE == "untargeted":
        target_lines = ["Untargeted reference source: each model's clean generation"]
    elif TARGET_LOSS_MODE == "standard_ce":
        target_lines = [f"Active target text: {configured_target_texts[0]}"]
    else:
        target_lines = ["Target texts:", *(f"- {target_text}" for target_text in configured_target_texts)]
    optimization_lines = [f"Cross-model optimization mode: {CROSS_MODEL_OPTIMIZATION_MODE}"]
    if CROSS_MODEL_OPTIMIZATION_MODE == "softminimax":
        optimization_lines.append(
            f"Cross-model softminimax temperature: {CROSS_MODEL_SOFTMINIMAX_TEMPERATURE}"
        )
    eot_lines = [f"EoT enabled: {USE_EOT}"]
    if USE_EOT:
        eot_lines.extend(
            [
                f"EoT train samples: {EOT_TRAIN_SAMPLES}",
                f"EoT evaluation samples: {EOT_EVAL_SAMPLES}",
                (
                    "EoT transforms: "
                    f"rotation={EOT_ROTATION_DEGREES}, "
                    f"perspective={EOT_PERSPECTIVE_DISTORTION}, "
                    f"crop_scale={EOT_CROP_SCALE}, "
                    f"crop_ratio={EOT_CROP_RATIO}, "
                    f"brightness={EOT_COLOR_JITTER_BRIGHTNESS}, "
                    f"contrast={EOT_COLOR_JITTER_CONTRAST}, "
                    f"saturation={EOT_COLOR_JITTER_SATURATION}, "
                    f"noise_std={EOT_GAUSSIAN_NOISE_STD}"
                ),
            ]
        )
    first, second = (target_lines, eot_lines) if target_first else (eot_lines, target_lines)
    lines = [f"Attack mode: {ATTACK_MODE}"]
    if ATTACK_MODE == "targeted":
        lines.append(f"Target loss mode: {TARGET_LOSS_MODE}")
    return [*lines, *optimization_lines, *first, *second]


def build_aggregate_summary_sections(
    clean_results: dict[str, dict],
    adv_results: dict[str, dict],
    *,
    clean_eot_summary: dict | None = None,
    adv_eot_summary: dict | None = None,
) -> list[list[str]]:
    metric_loss_label = get_metric_loss_label()
    clean_losses = {key: result["loss"] for key, result in clean_results.items()}
    adv_losses = {key: result["loss"] for key, result in adv_results.items()}
    worst_clean_key, worst_clean_loss, mean_clean_loss = summarize_loss_values(
        clean_losses,
        higher_is_worse=ATTACK_MODE == "targeted",
    )
    worst_adv_key, worst_adv_loss, mean_adv_loss = summarize_loss_values(
        adv_losses,
        higher_is_worse=ATTACK_MODE == "targeted",
    )

    sections: list[list[str]] = []
    if USE_EOT and clean_eot_summary is not None and adv_eot_summary is not None:
        sections.extend(
            [
                [
                    (
                        f"EoT worst-case clean {metric_loss_label} "
                        f"({clean_eot_summary['num_samples']} samples): "
                        f"{clean_eot_summary['worst_loss']:.6f}"
                    ),
                    f"EoT worst-case clean model: {clean_eot_summary['worst_key']}",
                    (
                        f"EoT worst-case adversarial {metric_loss_label} "
                        f"({adv_eot_summary['num_samples']} samples): "
                        f"{adv_eot_summary['worst_loss']:.6f}"
                    ),
                    f"EoT worst-case adversarial model: {adv_eot_summary['worst_key']}",
                ],
                [
                    f"EoT mean clean {metric_loss_label}: {clean_eot_summary['mean_loss']:.6f}",
                    f"EoT mean adversarial {metric_loss_label}: {adv_eot_summary['mean_loss']:.6f}",
                ],
            ]
        )
    sections.extend(
        [
            [
                f"Worst-case clean {metric_loss_label}: {worst_clean_loss:.6f}",
                f"Worst-case clean model: {worst_clean_key}",
                f"Worst-case adversarial {metric_loss_label}: {worst_adv_loss:.6f}",
                f"Worst-case adversarial model: {worst_adv_key}",
            ],
            [
                f"Mean clean {metric_loss_label}: {mean_clean_loss:.6f}",
                f"Mean adversarial {metric_loss_label}: {mean_adv_loss:.6f}",
            ],
        ]
    )
    return sections


def build_model_summary_lines(
    key: str,
    clean_results: dict[str, dict],
    adv_results: dict[str, dict],
    *,
    clean_eot_summary: dict | None = None,
    adv_eot_summary: dict | None = None,
) -> list[str]:
    metric_loss_label = get_metric_loss_label()
    lines = [
        f"{key} clean {metric_loss_label}: {clean_results[key]['loss']:.6f}",
        f"{key} adversarial {metric_loss_label}: {adv_results[key]['loss']:.6f}",
    ]
    return lines + (
        [
            f"{key} EoT mean clean {metric_loss_label}: {clean_eot_summary['per_model_mean_losses'][key]:.6f}",
            f"{key} EoT mean adversarial {metric_loss_label}: {adv_eot_summary['per_model_mean_losses'][key]:.6f}",
        ]
        if USE_EOT and clean_eot_summary is not None and adv_eot_summary is not None
        else []
    )


def build_report_lines(
    clean_results: dict[str, dict],
    adv_results: dict[str, dict],
    *,
    clean_eot_summary: dict | None = None,
    adv_eot_summary: dict | None = None,
) -> list[str]:
    aggregate_sections = build_aggregate_summary_sections(
        clean_results,
        adv_results,
        clean_eot_summary=clean_eot_summary,
        adv_eot_summary=adv_eot_summary,
    )
    lines = [
        f"Prompt: {USER_PROMPT}",
        *build_target_config_lines(get_configured_target_texts(), target_first=False),
        "",
        "",
        "Models:",
    ]
    lines.extend(
        f"- {model_spec['key']}: {model_spec['model_name']} on {canonicalize_cuda_device(model_spec['device'])}"
        for model_spec in MODEL_SPECS
    )
    for section in aggregate_sections:
        lines.extend(["", *section])
    lines.append("")

    for model_spec in MODEL_SPECS:
        key = model_spec["key"]
        lines.extend(
            [
                *build_model_summary_lines(
                    key,
                    clean_results,
                    adv_results,
                    clean_eot_summary=clean_eot_summary,
                    adv_eot_summary=adv_eot_summary,
                ),
                "",
                f"{key} clean generation:",
                clean_results[key]["generation"],
                "",
                f"{key} adversarial generation:",
                adv_results[key]["generation"],
                "",
            ]
        )
    lines.append(f"Final reusable adversarial image path: {OUTPUT_ADV_PATH.resolve()}")

    return lines


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Source image: {SOURCE_IMAGE_PATH}")
    print(f"Prompt: {USER_PROMPT}")

    validate_config()
    for line in build_target_config_lines(get_configured_target_texts(), target_first=True):
        print(f"[Info] {line}" if line.startswith("EoT transforms:") else line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_clean = load_image_tensor(SOURCE_IMAGE_PATH, torch.device("cpu"), ATTACK_IMAGE_SIZE)
    worker_config = {
        "user_prompt": USER_PROMPT,
        "attack_mode": ATTACK_MODE,
        "target_texts": TARGET_TEXTS,
        "target_loss_mode": TARGET_LOSS_MODE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_input_size": MODEL_INPUT_SIZE,
        "attack_image_size": ATTACK_IMAGE_SIZE,
        "clip_mean": CLIP_MEAN,
        "clip_std": CLIP_STD,
    }

    ctx = mp.get_context("spawn")
    workers: list[dict] = []
    try:
        workers = start_workers(ctx, MODEL_SPECS, worker_config)

        print("[Info] Evaluating clean image...")
        clean_results = evaluate_workers(workers, x_clean)
        if ATTACK_MODE == "untargeted":
            print("[Info] Initializing untargeted references from clean generations...")
            set_workers_untargeted_references(workers, clean_results)
            print("[Info] Re-evaluating clean image with untargeted reference loss...")
            clean_results = evaluate_workers(workers, x_clean)
        clean_eot_summary = None
        if USE_EOT:
            print("[Info] Evaluating clean image under EoT...")
            clean_eot_summary = evaluate_workers_eot(
                workers,
                x_clean,
                num_samples=EOT_EVAL_SAMPLES,
            )

        if USE_EOT:
            print("[Info] Starting multi-GPU AdamW text-generation attack with EoT...")
        else:
            print("[Info] Starting multi-GPU AdamW text-generation attack...")
        x_final, delta = run_attack(workers, x_clean)

        print("[Info] Evaluating adversarial image...")
        adv_results = evaluate_workers(workers, x_final)
        adv_eot_summary = None
        if USE_EOT:
            print("[Info] Evaluating adversarial image under EoT...")
            adv_eot_summary = evaluate_workers_eot(
                workers,
                x_final,
                num_samples=EOT_EVAL_SAMPLES,
            )

        transforms.ToPILImage()(x_final.squeeze(0).cpu()).save(OUTPUT_ADV_PATH)
        save_noise_visualization(delta, OUTPUT_NOISE_PATH)
        OUTPUT_REPORT_PATH.write_text(
            "\n".join(
                build_report_lines(
                    clean_results,
                    adv_results,
                    clean_eot_summary=clean_eot_summary,
                    adv_eot_summary=adv_eot_summary,
                )
            )
        )

        for section in build_aggregate_summary_sections(
            clean_results,
            adv_results,
            clean_eot_summary=clean_eot_summary,
            adv_eot_summary=adv_eot_summary,
        ):
            for line in section:
                print(f"[Info] {line}")
        for model_spec in MODEL_SPECS:
            key = model_spec["key"]
            for line in build_model_summary_lines(
                key,
                clean_results,
                adv_results,
                clean_eot_summary=clean_eot_summary,
                adv_eot_summary=adv_eot_summary,
            ):
                print(f"[Info] {line}")

        print(f"[Success] Saved adversarial image to {OUTPUT_ADV_PATH.resolve()}")
        print(f"[Success] Saved perturbation visualization to {OUTPUT_NOISE_PATH.resolve()}")
        print(f"[Success] Saved text report to {OUTPUT_REPORT_PATH.resolve()}")
        print(f"[Info] Reusable adversarial image path: {OUTPUT_ADV_PATH.resolve()}")
    finally:
        shutdown_workers(workers)


if __name__ == "__main__":
    main()
