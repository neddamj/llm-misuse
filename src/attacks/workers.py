import traceback

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from attacks.common import canonicalize_cuda_device
from attacks.prompting import (
    build_chat_prompt_inputs,
    build_target_batches,
    build_teacher_forced_batch,
    generate_greedy_text,
)
from attacks.vision import build_vision_inputs, resolve_model_family


def load_worker_state(model_spec: dict, worker_config: dict) -> dict:
    device_name = canonicalize_cuda_device(model_spec["device"])
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[Worker:{model_spec['key']}] Loading model {model_spec['model_name']} on {device_name}")
    processor = AutoProcessor.from_pretrained(model_spec["model_name"], use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_spec["model_name"],
        dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)

    model_family = resolve_model_family(model_spec["model_family"], model.config.model_type)
    prompt_processor_kwargs = {}
    if model_family == "qwen":
        vision_config = model.config.vision_config
        vision_state = {
            "device": device,
            "model_input_size": worker_config["model_input_size"],
            "patch_size": vision_config.patch_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            "merge_size": vision_config.spatial_merge_size,
            "mean": torch.tensor(worker_config["clip_mean"], device=device),
            "std": torch.tensor(worker_config["clip_std"], device=device),
            "dummy_image_size": (
                worker_config["model_input_size"],
                worker_config["model_input_size"],
            ),
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
    elif model_family == "llava_next":
        image_processor = processor.image_processor
        size = image_processor.size
        shortest_edge = size.get("shortest_edge")
        if shortest_edge is not None:
            resize_size = (int(shortest_edge), int(shortest_edge))
        else:
            resize_size = (
                int(size["height"]),
                int(size["width"]),
            )
        crop_size = (
            int(image_processor.crop_size["height"]),
            int(image_processor.crop_size["width"]),
        )
        vision_state = {
            "resize_size": resize_size,
            "crop_size": crop_size,
            "image_grid_pinpoints": [
                (int(height), int(width))
                for height, width in image_processor.image_grid_pinpoints
            ],
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (
                worker_config["attack_image_size"][1],
                worker_config["attack_image_size"][0],
            ),
        }
    elif model_family == "smolvlm":
        image_processor = processor.image_processor
        vision_state = {
            "resize_longest_edge": int(image_processor.size["longest_edge"]),
            "max_image_size": int(image_processor.max_image_size["longest_edge"]),
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (
                worker_config["attack_image_size"][1],
                worker_config["attack_image_size"][0],
            ),
        }
        prompt_processor_kwargs["do_image_splitting"] = False
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

    prompt_text, prompt_model_inputs = build_chat_prompt_inputs(
        processor,
        device,
        worker_config["user_prompt"],
        vision_state["dummy_image_size"],
        **prompt_processor_kwargs,
    )
    target_batches = (
        build_target_batches(
            processor.tokenizer,
            prompt_model_inputs,
            worker_config["target_texts"],
            device,
        )
        if worker_config["attack_mode"] == "targeted"
        else []
    )

    print(
        f"[Worker:{model_spec['key']}] Ready on {device_name} "
        f"(model_type={model.config.model_type}, family={model_family})"
    )

    return {
        "model_spec": model_spec,
        "device": device,
        "processor": processor,
        "model": model,
        "model_family": model_family,
        "vision_state": vision_state,
        "prompt_text": prompt_text,
        "prompt_model_inputs": prompt_model_inputs,
        "prompt_token_count": prompt_model_inputs["input_ids"].shape[1],
        "target_batches": target_batches,
        "target_loss_mode": worker_config["target_loss_mode"] if worker_config["attack_mode"] == "targeted" else None,
        "attack_mode": worker_config["attack_mode"],
        "max_new_tokens": worker_config["max_new_tokens"],
        "untargeted_reference_batch": None,
    }


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


def compute_target_pixel_values_grad(
    state: dict,
    target_batch: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pixel_values_ref = vision_inputs["pixel_values"].detach().requires_grad_(True)
    vision_inputs_ref = dict(vision_inputs)
    vision_inputs_ref["pixel_values"] = pixel_values_ref
    score = target_score(state, target_batch, vision_inputs_ref)
    if weight is not None:
        score = weight * score
    return torch.autograd.grad(-score, pixel_values_ref)[0]


def compute_untargeted_reference_pixel_values_grad(
    state: dict,
    reference_batch: dict,
    vision_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    pixel_values_ref = vision_inputs["pixel_values"].detach().requires_grad_(True)
    vision_inputs_ref = dict(vision_inputs)
    vision_inputs_ref["pixel_values"] = pixel_values_ref
    score = target_score(state, reference_batch, vision_inputs_ref)
    return torch.autograd.grad(score, pixel_values_ref)[0]


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

        grad = compute_target_pixel_values_grad(state, target_batch, vision_inputs)
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
        pixel_values_grad.add_(
            compute_target_pixel_values_grad(
                state,
                target_batch,
                vision_inputs,
                weight=weight,
            )
        )

    vision_inputs["pixel_values"].backward(pixel_values_grad)
    return aggregate_loss


def untargeted_reference_loss(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_batch = state["untargeted_reference_batch"]
    if reference_batch is None:
        raise RuntimeError("Untargeted mode requires a clean reference generation before evaluation or attack.")

    with torch.no_grad():
        metric_loss = -target_score(state, reference_batch, vision_inputs)
        optimization_loss = -metric_loss

    if not backward:
        return metric_loss, optimization_loss

    grad = compute_untargeted_reference_pixel_values_grad(state, reference_batch, vision_inputs)
    vision_inputs["pixel_values"].backward(grad)
    return metric_loss, optimization_loss


def attack_loss(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state["attack_mode"] == "targeted":
        metric_loss = target_loss(state, vision_inputs, backward=backward)
        return metric_loss, metric_loss
    return untargeted_reference_loss(state, vision_inputs, backward=backward)


def generate_from_image(state: dict, image_tensor: torch.Tensor) -> str:
    vision_inputs = build_vision_inputs(state, image_tensor)
    return generate_greedy_text(
        state["model"],
        state["processor"],
        state["prompt_model_inputs"],
        state["prompt_token_count"],
        vision_inputs,
        max_new_tokens=state["max_new_tokens"],
    )


def evaluate_image(state: dict, image_cpu: torch.Tensor) -> dict:
    image_gpu = image_cpu.to(state["device"], non_blocking=True).squeeze(0)
    with torch.no_grad():
        generation = generate_from_image(state, image_gpu)
        loss = None
        if state["attack_mode"] == "targeted" or state["untargeted_reference_batch"] is not None:
            vision_inputs = build_vision_inputs(state, image_gpu)
            metric_loss, _ = attack_loss(state, vision_inputs)
            loss = float(metric_loss.item())
    return {
        "loss": loss,
        "generation": generation,
    }


def attack_step(state: dict, image_cpu: torch.Tensor) -> dict:
    x_adv = image_cpu.to(state["device"], non_blocking=True).detach().clone().requires_grad_(True)
    vision_inputs = build_vision_inputs(state, x_adv.squeeze(0))
    metric_loss, optimization_loss = attack_loss(state, vision_inputs, backward=True)
    if x_adv.grad is None:
        raise RuntimeError("Expected a gradient on the adversarial image tensor.")
    return {
        "loss": float(metric_loss.item()),
        "optimization_loss": float(optimization_loss.item()),
        "grad": x_adv.grad.detach().cpu(),
    }


def worker_main(model_spec: dict, worker_config: dict, request_queue, response_queue) -> None:
    try:
        state = load_worker_state(model_spec, worker_config)
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
            if command == "set_untargeted_reference":
                reference_text = message["reference_text"]
                if not reference_text:
                    raise RuntimeError(
                        f"Untargeted mode requires a non-empty clean generation for {model_spec['key']}."
                    )
                state["untargeted_reference_batch"] = {
                    "reference_text": reference_text,
                    **build_teacher_forced_batch(
                        state["processor"].tokenizer,
                        state["prompt_model_inputs"],
                        reference_text,
                        state["device"],
                    ),
                }
                response_queue.put({"type": "set_untargeted_reference", "key": model_spec["key"]})
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


def dispatch_worker_command(
    workers: list[dict],
    command: str,
    image_cpu: torch.Tensor,
    *,
    expected_type: str,
) -> list[dict]:
    image_cpu = image_cpu.detach().cpu().contiguous()
    for worker in workers:
        worker["request_queue"].put({"command": command, "image": image_cpu})
    messages = [receive_message(worker) for worker in workers]
    for worker, message in zip(workers, messages):
        if message["type"] != expected_type:
            raise RuntimeError(
                f"Expected {expected_type} result from {worker['model_spec']['key']}, "
                f"got {message['type']!r}."
            )
    return messages


def start_workers(ctx, model_specs: list[dict], worker_config: dict) -> list[dict]:
    workers = []
    for model_spec in model_specs:
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        process = ctx.Process(
            target=worker_main,
            args=(model_spec, worker_config, request_queue, response_queue),
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
    return {
        message["key"]: {
            "loss": message["loss"],
            "generation": message["generation"],
        }
        for message in dispatch_worker_command(workers, "evaluate", image_cpu, expected_type="evaluate")
    }


def attack_workers(
    workers: list[dict],
    image_cpu: torch.Tensor,
) -> tuple[list[str], dict[str, float], dict[str, float], list[torch.Tensor]]:
    messages = dispatch_worker_command(workers, "attack_step", image_cpu, expected_type="attack_step")
    return (
        [message["key"] for message in messages],
        {message["key"]: message["loss"] for message in messages},
        {message["key"]: message["optimization_loss"] for message in messages},
        [message["grad"] for message in messages],
    )


def set_workers_untargeted_references(
    workers: list[dict],
    clean_results: dict[str, dict],
) -> None:
    for worker in workers:
        key = worker["model_spec"]["key"]
        reference_text = clean_results[key]["generation"]
        if not reference_text:
            raise RuntimeError(f"Untargeted mode requires a non-empty clean generation for {key}.")
        worker["request_queue"].put(
            {
                "command": "set_untargeted_reference",
                "reference_text": reference_text,
            }
        )

    messages = [receive_message(worker) for worker in workers]
    for worker, message in zip(workers, messages):
        if message["type"] != "set_untargeted_reference":
            raise RuntimeError(
                f"Expected set_untargeted_reference result from {worker['model_spec']['key']}, "
                f"got {message['type']!r}."
            )
