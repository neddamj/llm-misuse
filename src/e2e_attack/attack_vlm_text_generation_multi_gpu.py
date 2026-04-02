from pathlib import Path
import traceback

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo root from the current working directory. "
        "Launch the script from this repository or one of its subdirectories."
    )


REPO_ROOT = find_repo_root()
RESULTS_DIR = REPO_ROOT / "results"

MODEL_SPECS = [
    {
        "key": "qwen3_vl",
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "model_family": "auto",
        "device": "cuda:0",
    },
    {
        "key": "llava",
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "model_family": "auto",
        "device": "cuda:1",
    },
    {
        "key": "gemma3_4b_it",
        "model_name": "google/gemma-3-4b-it",
        "model_family": "auto",
        "device": "cuda:2",
    },
]

SOURCE_IMAGE_PATH = REPO_ROOT / "data" / "images" / "worksheet_000002.png"
USER_PROMPT = "Solve the math problems shown in the image. Show steps and give the final answers."
TARGET_TEXTS: list[str] = [
    "I cannot help with that",
    "No",
    "I'm sorry, I cant answer that.",
]
TARGET_LOSS_MODE = "standard_ce" # Options: "standard_ce", "multi_reference"

EPSILON = 64 / 255
ALPHA = 1 / 255
STEPS = 2000
ATTACK_IMAGE_SIZE = (400, 400)
MODEL_INPUT_SIZE = 448
MAX_NEW_TOKENS = 128

RESULT_PREFIX = "qwen3vl_llava_gemma3_textgen_multi_gpu"
OUTPUT_ADV_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_adv.png"
OUTPUT_NOISE_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_noise.png"
OUTPUT_REPORT_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_generations.txt"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

SUPPORTED_MODEL_TYPES = {
    "gemma3": "gemma",
    "qwen2_vl": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen3_vl": "qwen",
    "llava": "llava",
}

TOKEN_TYPE_INPUT_KEYS = ("mm_token_type_ids", "token_type_ids")
SUPPORTED_TARGET_LOSS_MODES = {"multi_reference", "standard_ce"}


def load_image_tensor(
    image_path: str | Path,
    device: torch.device,
    image_size: tuple[int, int],
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BICUBIC)
    return transforms.ToTensor()(image).to(device).unsqueeze(0)


def resolve_model_family(requested_model_family: str, model_type: str) -> str:
    if requested_model_family not in {"auto", "gemma", "qwen", "llava"}:
        raise ValueError("MODEL_FAMILY must be one of: 'auto', 'gemma', 'qwen', 'llava'.")

    detected_model_family = SUPPORTED_MODEL_TYPES.get(model_type)
    if detected_model_family is None:
        supported_model_types = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unsupported model type {model_type!r}. "
            f"This script supports model types: {supported_model_types}."
        )

    if requested_model_family == "auto":
        return detected_model_family

    if requested_model_family != detected_model_family:
        raise ValueError(
            f"MODEL_FAMILY={requested_model_family!r} does not match model type {model_type!r}. "
            f"Use MODEL_FAMILY={detected_model_family!r} or 'auto'."
        )

    return requested_model_family


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
    )
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=device, dtype=torch.long)
    return pixel_values, image_grid_thw


def pack_for_llava(
    image_tensor: torch.Tensor,
    *,
    shortest_edge: int,
    crop_size: tuple[int, int],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    height, width = image_tensor.shape[-2:]
    scale = shortest_edge / min(height, width)
    resized_h = max(crop_size[0], int(round(height * scale)))
    resized_w = max(crop_size[1], int(round(width * scale)))

    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )

    crop_h, crop_w = crop_size
    top = max(0, (resized_h - crop_h) // 2)
    left = max(0, (resized_w - crop_w) // 2)
    x = x[:, :, top : top + crop_h, left : left + crop_w]
    return (x - mean) / std


def pack_for_gemma(
    image_tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    if torch.max(x) > 1.0:
        x = x * rescale_factor

    return (x - mean) / std


def save_noise_visualization(delta: torch.Tensor, output_path: Path) -> None:
    noise = torch.clamp(delta.squeeze(0).cpu() * 10 + 0.5, 0.0, 1.0)
    transforms.ToPILImage()(noise).save(output_path)


def canonicalize_cuda_device(device_name: str) -> str:
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None:
        raise ValueError(f"Expected an explicit CUDA device like 'cuda:0', got {device_name!r}.")
    return f"cuda:{device.index}"


def validate_config() -> None:
    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE_PATH}")

    if TARGET_LOSS_MODE not in SUPPORTED_TARGET_LOSS_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_TARGET_LOSS_MODES))
        raise ValueError(
            f"TARGET_LOSS_MODE must be one of: {supported_modes}. "
            f"Got {TARGET_LOSS_MODE!r}."
        )

    if not TARGET_TEXTS or any(not isinstance(target_text, str) or not target_text for target_text in TARGET_TEXTS):
        raise ValueError("TARGET_TEXTS must be a non-empty list of non-empty strings.")

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
    dummy_image_size: tuple[int, int],
    prompt: str,
    **processor_kwargs,
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
    dummy_image = Image.new("RGB", dummy_image_size, color="white")
    prompt_inputs = processor(
        text=[rendered_prompt],
        images=[dummy_image],
        return_tensors="pt",
        **processor_kwargs,
    )
    return rendered_prompt, build_text_model_inputs(prompt_inputs, device)


def build_target_batches(
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    target_texts: list[str],
    device: torch.device,
) -> list[dict]:
    eos_token_id = tokenizer.eos_token_id
    target_batches = []
    for target_text in target_texts:
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        if eos_token_id is not None and (target_ids.shape[1] == 0 or target_ids[0, -1].item() != eos_token_id):
            target_ids = torch.cat([target_ids, torch.tensor([[eos_token_id]], device=device)], dim=1)

        full_model_inputs = {
            "input_ids": torch.cat([prompt_model_inputs["input_ids"], target_ids], dim=1),
            "attention_mask": torch.cat([prompt_model_inputs["attention_mask"], torch.ones_like(target_ids)], dim=1),
        }
        for token_type_key in TOKEN_TYPE_INPUT_KEYS:
            token_type_ids = prompt_model_inputs.get(token_type_key)
            if token_type_ids is not None:
                full_model_inputs[token_type_key] = torch.cat(
                    [
                        token_type_ids,
                        torch.zeros(
                            target_ids.shape,
                            device=device,
                            dtype=token_type_ids.dtype,
                        ),
                    ],
                    dim=1,
                )

        labels = full_model_inputs["input_ids"].clone()
        labels[:, : prompt_model_inputs["input_ids"].shape[1]] = -100
        target_batches.append(
            {
                "target_text": target_text,
                "model_inputs": full_model_inputs,
                "labels": labels,
            }
        )

    return target_batches


def get_configured_target_texts() -> list[str]:
    if TARGET_LOSS_MODE == "standard_ce":
        return [TARGET_TEXTS[0]]
    return TARGET_TEXTS


def load_worker_state(model_spec: dict) -> dict:
    device_name = canonicalize_cuda_device(model_spec["device"])
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[Worker:{model_spec['key']}] Loading model {model_spec['model_name']} on {device_name}")
    processor = AutoProcessor.from_pretrained(model_spec["model_name"], use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_spec["model_name"],
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)

    model_family = resolve_model_family(model_spec["model_family"], model.config.model_type)
    prompt_processor_kwargs = {}
    if model_family == "qwen":
        vision_config = model.config.vision_config
        vision_state = {
            "patch_size": vision_config.patch_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            "merge_size": vision_config.spatial_merge_size,
            "mean": torch.tensor(CLIP_MEAN, device=device),
            "std": torch.tensor(CLIP_STD, device=device),
            "dummy_image_size": (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        }
    elif model_family == "gemma":
        image_processor = processor.image_processor
        size = (
            int(image_processor.size["height"]),
            int(image_processor.size["width"]),
        )
        vision_state = {
            "size": size,
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (size[1], size[0]),
        }
        prompt_processor_kwargs["do_pan_and_scan"] = False
    else:
        image_processor = processor.image_processor
        shortest_edge = image_processor.size.get("shortest_edge")
        if shortest_edge is None:
            shortest_edge = min(image_processor.size["height"], image_processor.size["width"])
        crop_size = (
            int(image_processor.crop_size["height"]),
            int(image_processor.crop_size["width"]),
        )
        vision_state = {
            "shortest_edge": int(shortest_edge),
            "crop_size": crop_size,
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (crop_size[1], crop_size[0]),
        }

    prompt_text, prompt_model_inputs = build_prompt_inputs(
        processor,
        device,
        vision_state["dummy_image_size"],
        USER_PROMPT,
        **prompt_processor_kwargs,
    )
    prompt_input_ids = prompt_model_inputs["input_ids"]
    target_batches = build_target_batches(
        processor.tokenizer,
        prompt_model_inputs,
        TARGET_TEXTS,
        device,
    )

    print(
        f"[Worker:{model_spec['key']}] Ready on {device_name} "
        f"(model_type={model.config.model_type}, family={model_family})"
    )

    return {
        "model_spec": model_spec,
        "device": device,
        "dtype": dtype,
        "processor": processor,
        "model": model,
        "model_family": model_family,
        "vision_state": vision_state,
        "prompt_text": prompt_text,
        "prompt_model_inputs": prompt_model_inputs,
        "prompt_input_ids": prompt_input_ids,
        "target_batches": target_batches,
        "target_loss_mode": TARGET_LOSS_MODE,
    }


def build_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    if state["model_family"] == "qwen":
        vision_state = state["vision_state"]
        pixel_values, image_grid_thw = pack_for_qwen(
            image_tensor,
            model_input_size=MODEL_INPUT_SIZE,
            mean=vision_state["mean"].view(3, 1, 1),
            std=vision_state["std"].view(3, 1, 1),
            patch_size=vision_state["patch_size"],
            temporal_patch_size=vision_state["temporal_patch_size"],
            merge_size=vision_state["merge_size"],
            device=state["device"],
        )
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    if state["model_family"] == "gemma":
        vision_state = state["vision_state"]
        pixel_values = pack_for_gemma(
            image_tensor,
            size=vision_state["size"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
        )
        return {"pixel_values": pixel_values}

    vision_state = state["vision_state"]
    pixel_values = pack_for_llava(
        image_tensor,
        shortest_edge=vision_state["shortest_edge"],
        crop_size=vision_state["crop_size"],
        mean=vision_state["mean"].view(1, 3, 1, 1),
        std=vision_state["std"].view(1, 3, 1, 1),
    )
    return {"pixel_values": pixel_values}


def target_score(state: dict, target_batch: dict, vision_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = state["model"](
        **target_batch["model_inputs"],
        **vision_inputs,
        use_cache=False,
        return_dict=True,
    )
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = target_batch["labels"][:, 1:]
    valid_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_mask = valid_mask.to(token_log_probs.dtype)
    avg_nll = -(token_log_probs * token_mask).sum() / token_mask.sum()
    return -avg_nll


def target_loss(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> torch.Tensor:
    if state["target_loss_mode"] == "standard_ce":
        target_batch = state["target_batches"][0]
        with torch.no_grad():
            loss = -target_score(state, target_batch, vision_inputs)

        if not backward:
            return loss

        pixel_values_ref = vision_inputs["pixel_values"].detach().requires_grad_(True)
        vision_inputs_ref = dict(vision_inputs)
        vision_inputs_ref["pixel_values"] = pixel_values_ref
        score = target_score(state, target_batch, vision_inputs_ref)
        grad = torch.autograd.grad(-score, pixel_values_ref)[0]
        vision_inputs["pixel_values"].backward(grad)
        return loss

    with torch.no_grad():
        detached_scores = torch.stack(
            [target_score(state, target_batch, vision_inputs) for target_batch in state["target_batches"]]
        )
        aggregate_loss = -(
            torch.logsumexp(detached_scores, dim=0) - detached_scores.new_tensor(len(state["target_batches"])).log()
        )

    if not backward:
        return aggregate_loss

    weights = torch.softmax(detached_scores, dim=0)
    pixel_values_grad = torch.zeros_like(vision_inputs["pixel_values"])
    for weight, target_batch in zip(weights, state["target_batches"]):
        pixel_values_ref = vision_inputs["pixel_values"].detach().requires_grad_(True)
        vision_inputs_ref = dict(vision_inputs)
        vision_inputs_ref["pixel_values"] = pixel_values_ref
        score = target_score(state, target_batch, vision_inputs_ref)
        grad = torch.autograd.grad(-weight * score, pixel_values_ref)[0]
        pixel_values_grad.add_(grad)

    vision_inputs["pixel_values"].backward(pixel_values_grad)
    return aggregate_loss


def generate_from_image(state: dict, image_tensor: torch.Tensor) -> str:
    vision_inputs = build_vision_inputs(state, image_tensor)
    generated = state["model"].generate(
        **state["prompt_model_inputs"],
        **vision_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    new_tokens = generated[:, state["prompt_input_ids"].shape[1] :]
    return state["processor"].batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def evaluate_image(state: dict, image_cpu: torch.Tensor) -> dict:
    image_gpu = image_cpu.to(state["device"], non_blocking=True).squeeze(0)
    with torch.no_grad():
        vision_inputs = build_vision_inputs(state, image_gpu)
        loss = float(target_loss(state, vision_inputs).item())
        generation = generate_from_image(state, image_gpu)
    return {
        "loss": loss,
        "generation": generation,
    }


def attack_step(state: dict, image_cpu: torch.Tensor) -> dict:
    x_adv = image_cpu.to(state["device"], non_blocking=True).detach().clone().requires_grad_(True)
    vision_inputs = build_vision_inputs(state, x_adv.squeeze(0))
    loss = target_loss(state, vision_inputs, backward=True)
    if x_adv.grad is None:
        raise RuntimeError("Expected a gradient on the adversarial image tensor.")
    return {
        "loss": float(loss.item()),
        "grad": x_adv.grad.detach().cpu(),
    }


def worker_main(model_spec: dict, request_queue, response_queue) -> None:
    try:
        state = load_worker_state(model_spec)
        response_queue.put(
            {
                "type": "ready",
                "key": model_spec["key"],
                "model_name": model_spec["model_name"],
                "device": canonicalize_cuda_device(model_spec["device"]),
                "prompt_text": state["prompt_text"],
            }
        )

        while True:
            message = request_queue.get()
            command = message["command"]
            if command == "shutdown":
                response_queue.put({"type": "shutdown", "key": model_spec["key"]})
                return
            if command == "attack_step":
                result = attack_step(state, message["image"])
                response_queue.put({"type": "attack_step", "key": model_spec["key"], **result})
                continue
            if command == "evaluate":
                result = evaluate_image(state, message["image"])
                response_queue.put({"type": "evaluate", "key": model_spec["key"], **result})
                continue
            raise ValueError(f"Unsupported command: {command!r}")
    except Exception:
        response_queue.put(
            {
                "type": "error",
                "key": model_spec["key"],
                "message": traceback.format_exc(),
            }
        )


def receive_message(worker: dict) -> dict:
    message = worker["response_queue"].get()
    if message["type"] == "error":
        raise RuntimeError(
            f"Worker {worker['model_spec']['key']} failed:\n{message['message']}"
        )
    return message


def start_workers(ctx) -> list[dict]:
    workers = []
    for model_spec in MODEL_SPECS:
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        process = ctx.Process(
            target=worker_main,
            args=(model_spec, request_queue, response_queue),
        )
        process.start()
        workers.append(
            {
                "model_spec": model_spec,
                "process": process,
                "request_queue": request_queue,
                "response_queue": response_queue,
            }
        )

    ready_messages = {}
    for worker in workers:
        message = receive_message(worker)
        if message["type"] != "ready":
            raise RuntimeError(
                f"Expected worker {worker['model_spec']['key']} to send a ready message, "
                f"got {message['type']!r}."
            )
        ready_messages[message["key"]] = message

    print("[Info] Started model workers:")
    for worker in workers:
        ready = ready_messages[worker["model_spec"]["key"]]
        print(f"- {ready['key']}: {ready['model_name']} on {ready['device']}")

    return workers


def shutdown_workers(workers: list[dict]) -> None:
    for worker in workers:
        if worker["process"].is_alive():
            worker["request_queue"].put({"command": "shutdown"})

    for worker in workers:
        if worker["process"].is_alive():
            try:
                receive_message(worker)
            except Exception:
                pass

    for worker in workers:
        worker["process"].join(timeout=5)
        if worker["process"].is_alive():
            worker["process"].terminate()
            worker["process"].join(timeout=5)


def evaluate_workers(workers: list[dict], image_cpu: torch.Tensor) -> dict[str, dict]:
    image_cpu = image_cpu.detach().cpu().contiguous()
    for worker in workers:
        worker["request_queue"].put({"command": "evaluate", "image": image_cpu})

    results = {}
    for worker in workers:
        message = receive_message(worker)
        if message["type"] != "evaluate":
            raise RuntimeError(
                f"Expected evaluate result from {worker['model_spec']['key']}, got {message['type']!r}."
            )
        results[message["key"]] = {
            "loss": message["loss"],
            "generation": message["generation"],
        }

    return results


def run_attack(workers: list[dict], x_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.zeros_like(x_clean, requires_grad=True)
    optimizer = torch.optim.AdamW([delta], lr=ALPHA, weight_decay=0.0)
    progress = tqdm(range(STEPS))

    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach().cpu().contiguous()

        for worker in workers:
            worker["request_queue"].put({"command": "attack_step", "image": x_adv})

        step_results = {}
        gradients = []
        for worker in workers:
            message = receive_message(worker)
            if message["type"] != "attack_step":
                raise RuntimeError(
                    f"Expected attack_step result from {worker['model_spec']['key']}, got {message['type']!r}."
                )
            step_results[message["key"]] = message["loss"]
            gradients.append(message["grad"])

        mean_grad = torch.stack(gradients, dim=0).mean(dim=0)
        delta.grad = mean_grad.detach()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-EPSILON, EPSILON)
            delta.copy_(torch.clamp(x_clean + delta, 0.0, 1.0) - x_clean)

        mean_loss = sum(step_results.values()) / len(step_results)
        progress.set_postfix(
            {
                f"{key}_loss": f"{value:.4f}"
                for key, value in step_results.items()
            }
            | {"mean_loss": f"{mean_loss:.4f}"}
        )

    return torch.clamp(x_clean + delta, 0.0, 1.0).detach(), delta.detach()


def build_report_lines(
    clean_results: dict[str, dict],
    adv_results: dict[str, dict],
) -> list[str]:
    mean_clean_loss = sum(result["loss"] for result in clean_results.values()) / len(clean_results)
    mean_adv_loss = sum(result["loss"] for result in adv_results.values()) / len(adv_results)
    configured_target_texts = get_configured_target_texts()

    lines = [
        f"Prompt: {USER_PROMPT}",
        f"Target loss mode: {TARGET_LOSS_MODE}",
    ]
    if TARGET_LOSS_MODE == "standard_ce":
        lines.extend(
            [
                f"Active target text: {configured_target_texts[0]}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Target texts:",
                *[f"- {target_text}" for target_text in configured_target_texts],
                "",
            ]
        )

    lines.extend(
        [
            "",
            "Models:",
        ]
    )
    for model_spec in MODEL_SPECS:
        lines.append(
            f"- {model_spec['key']}: {model_spec['model_name']} on {canonicalize_cuda_device(model_spec['device'])}"
        )

    lines.extend(
        [
            "",
            f"Mean clean target loss: {mean_clean_loss:.6f}",
            f"Mean adversarial target loss: {mean_adv_loss:.6f}",
            "",
        ]
    )

    for model_spec in MODEL_SPECS:
        key = model_spec["key"]
        lines.extend(
            [
                f"{key} clean target loss: {clean_results[key]['loss']:.6f}",
                f"{key} adversarial target loss: {adv_results[key]['loss']:.6f}",
                "",
                f"{key} clean generation:",
                clean_results[key]["generation"],
                "",
                f"{key} adversarial generation:",
                adv_results[key]["generation"],
                "",
            ]
        )

    lines.extend(
        [
            f"Final reusable adversarial image path: {OUTPUT_ADV_PATH.resolve()}",
        ]
    )

    return lines


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Source image: {SOURCE_IMAGE_PATH}")
    print(f"Prompt: {USER_PROMPT}")

    validate_config()
    configured_target_texts = get_configured_target_texts()

    print(f"Target loss mode: {TARGET_LOSS_MODE}")
    if TARGET_LOSS_MODE == "standard_ce":
        print(f"Active target text: {configured_target_texts[0]}")
    else:
        print("Target texts:")
        for target_text in configured_target_texts:
            print(f"- {target_text}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_clean = load_image_tensor(SOURCE_IMAGE_PATH, torch.device("cpu"), ATTACK_IMAGE_SIZE)

    ctx = mp.get_context("spawn")
    workers: list[dict] = []
    try:
        workers = start_workers(ctx)

        print("[Info] Evaluating clean image...")
        clean_results = evaluate_workers(workers, x_clean)

        print("[Info] Starting multi-GPU AdamW text-generation attack...")
        x_final, delta = run_attack(workers, x_clean)

        print("[Info] Evaluating adversarial image...")
        adv_results = evaluate_workers(workers, x_final)

        transforms.ToPILImage()(x_final.squeeze(0).cpu()).save(OUTPUT_ADV_PATH)
        save_noise_visualization(delta, OUTPUT_NOISE_PATH)
        OUTPUT_REPORT_PATH.write_text("\n".join(build_report_lines(clean_results, adv_results)))

        mean_clean_loss = sum(result["loss"] for result in clean_results.values()) / len(clean_results)
        mean_adv_loss = sum(result["loss"] for result in adv_results.values()) / len(adv_results)

        print(f"[Info] Mean clean target loss: {mean_clean_loss:.6f}")
        print(f"[Info] Mean adversarial target loss: {mean_adv_loss:.6f}")
        for model_spec in MODEL_SPECS:
            key = model_spec["key"]
            print(f"[Info] {key} clean target loss: {clean_results[key]['loss']:.6f}")
            print(f"[Info] {key} adversarial target loss: {adv_results[key]['loss']:.6f}")

        print(f"[Success] Saved adversarial image to {OUTPUT_ADV_PATH.resolve()}")
        print(f"[Success] Saved perturbation visualization to {OUTPUT_NOISE_PATH.resolve()}")
        print(f"[Success] Saved text report to {OUTPUT_REPORT_PATH.resolve()}")
        print(f"[Info] Reusable adversarial image path: {OUTPUT_ADV_PATH.resolve()}")
    finally:
        shutdown_workers(workers)


if __name__ == "__main__":
    main()
