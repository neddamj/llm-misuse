import queue
import sys
import traceback
import time
from typing import TypedDict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    InstructBlipProcessor,
)

from attacks.common import canonicalize_cuda_device
from attacks.prompting import (
    build_chat_prompt_inputs,
    build_target_batches,
    build_teacher_forced_batch,
    generate_greedy_text,
)
from attacks.vision import build_vision_inputs, resolve_model_family


class _WorkerTee:
    """Mirror spawned-worker output into the parent run's terminal log."""

    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.stream, name)


def ensure_remote_processor_compat(model_name: str, revision: str | None = None) -> None:
    # Keep remote processor/model code working across transformer/numpy version mismatches.
    import importlib
    import numpy as np
    import transformers.modeling_rope_utils as rope_utils
    import transformers.processing_utils as processing_utils
    from transformers import AutoConfig

    if not hasattr(processing_utils, "CommonKwargs"):
        # Some remote processors still import this older typing alias.
        class CommonKwargs(TypedDict, total=False):
            pass

        processing_utils.CommonKwargs = CommonKwargs

    if not hasattr(np, "concat"):
        # Backfill alias expected by some remote repos.
        np.concat = np.concatenate

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        # Remote rotary implementations can expect a "default" rope initializer to exist.
        def default_rope_init_fn(config, device=None, seq_len=None, layer_type=None):
            base = float(getattr(config, "rope_theta"))
            partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                num_attention_heads = getattr(config, "num_attention_heads", None)
                if num_attention_heads is None:
                    num_attention_heads = getattr(config, "n_heads")
                head_dim = getattr(config, "hidden_size") // num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device,
                        dtype=torch.float,
                    )
                    / dim
                )
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope_init_fn

    if model_name == "jinaai/jina-vlm":
        revision_kwargs = {"revision": revision} if revision else {}
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            **revision_kwargs,
        )
        module_base = config.__class__.__module__.rsplit(".", 1)[0]
        blocks_module = importlib.import_module(f"{module_base}.blocks_jvlm")
        rotary_embedding_cls = blocks_module.RotaryEmbedding

        if not hasattr(rotary_embedding_cls, "compute_default_rope_parameters"):
            def compute_default_rope_parameters(
                self,
                config=None,
                device=None,
                seq_len=None,
                layer_type=None,
            ):
                if config is None:
                    config = self.config
                return default_rope_init_fn(
                    config,
                    device=device,
                    seq_len=seq_len,
                    layer_type=layer_type,
                )

            rotary_embedding_cls.compute_default_rope_parameters = compute_default_rope_parameters


def load_worker_state(model_spec: dict, worker_config: dict) -> dict:
    device_name = canonicalize_cuda_device(model_spec["device"])
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[Worker:{model_spec['key']}] Loading model {model_spec['model_name']} on {device_name}")
    trust_remote_code = bool(model_spec.get("trust_remote_code", False))
    revision = model_spec.get("revision")
    revision_kwargs = {"revision": revision} if revision else {}
    if trust_remote_code:
        ensure_remote_processor_compat(model_spec["model_name"], revision)
    try:
        processor = AutoProcessor.from_pretrained(
            model_spec["model_name"],
            use_fast=False,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        )
    except ValueError as exc:
        if "does not have a slow version" not in str(exc):
            raise
        processor = AutoProcessor.from_pretrained(
            model_spec["model_name"],
            use_fast=True,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        )

    if (
        model_spec.get("model_family") == "instructblip"
        and not hasattr(processor, "image_processor")
    ):
        try:
            processor = InstructBlipProcessor.from_pretrained(
                model_spec["model_name"],
                trust_remote_code=trust_remote_code,
                **revision_kwargs,
            )
        except ValueError as exc:
            if "does not have a slow version" not in str(exc):
                raise
            processor = InstructBlipProcessor.from_pretrained(
                model_spec["model_name"],
                use_fast=True,
                trust_remote_code=trust_remote_code,
                **revision_kwargs,
            )

    auto_model_class = model_spec.get("auto_model_class", "image_text_to_text")
    if auto_model_class == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_spec["model_name"],
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        ).to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_spec["model_name"],
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **revision_kwargs,
        ).to(device)
    model.eval()
    model.requires_grad_(False)

    model_family = resolve_model_family(model_spec["model_family"], model.config.model_type)
    prompt_processor_kwargs = {}
    vision_grad_input_key = "pixel_values"
    if model_family == "aya_vision":
        image_processor = processor.image_processor
        tile_size = (
            int(image_processor.size["height"]),
            int(image_processor.size["width"]),
        )
        vision_state = {
            "tile_size": tile_size,
            "min_patches": int(getattr(image_processor, "min_patches", 1)),
            "max_patches": int(getattr(image_processor, "max_patches", 12)),
            "rescale_factor": float(getattr(image_processor, "rescale_factor", 1 / 255)),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "use_thumbnail": True,
            "dummy_image_size": (
                worker_config["attack_image_size"][1],
                worker_config["attack_image_size"][0],
            ),
        }
        prompt_processor_kwargs["return_mm_token_type_ids"] = True
    elif model_family == "qwen":
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
    elif model_family == "instructblip":
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
    elif model_family == "jina_vlm":
        image_processor = processor.image_processor
        base_input_size = getattr(image_processor, "base_input_size", (378, 378))
        if isinstance(base_input_size, list):
            base_input_size = tuple(int(value) for value in base_input_size)
        vision_state = {
            "min_pixels": int(getattr(image_processor, "min_pixels")),
            "max_pixels": int(getattr(image_processor, "max_pixels")),
            "patch_size": int(getattr(image_processor, "patch_size")),
            "max_crops": int(getattr(image_processor, "max_crops")),
            "base_input_size": (
                int(base_input_size[0]),
                int(base_input_size[1]),
            ),
            "overlap_margins": tuple(int(value) for value in getattr(image_processor, "overlap_margins")),
            "pooling_w": int(getattr(image_processor, "pooling_w")),
            "pooling_h": int(getattr(image_processor, "pooling_h")),
            "token_length_w": int(getattr(image_processor, "token_length_w")),
            "token_length_h": int(getattr(image_processor, "token_length_h")),
            "use_column_tokens": bool(getattr(image_processor, "use_column_tokens")),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (
                worker_config["attack_image_size"][1],
                worker_config["attack_image_size"][0],
            ),
        }
        prompt_processor_kwargs["return_mm_token_type_ids"] = True
        vision_grad_input_key = "image_patches"
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
    elif model_family == "lfm2_vl":
        image_processor = processor.image_processor
        vision_state = {
            "downsample_factor": int(image_processor.downsample_factor),
            "do_image_splitting": bool(image_processor.do_image_splitting),
            "min_tiles": int(image_processor.min_tiles),
            "max_tiles": int(image_processor.max_tiles),
            "use_thumbnail": bool(image_processor.use_thumbnail),
            "min_image_tokens": int(image_processor.min_image_tokens),
            "max_image_tokens": int(image_processor.max_image_tokens),
            "encoder_patch_size": int(image_processor.encoder_patch_size),
            "tile_size": int(image_processor.tile_size),
            "max_pixels_tolerance": float(image_processor.max_pixels_tolerance),
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": torch.tensor(image_processor.image_mean, device=device),
            "std": torch.tensor(image_processor.image_std, device=device),
            "dummy_image_size": (
                worker_config["attack_image_size"][1],
                worker_config["attack_image_size"][0],
            ),
        }
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
    if model_family == "gemma":
        torch_version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
        if torch_version < (2, 6):
            # Gemma 3 token_type_ids trigger Transformers' image bidirectional mask path,
            # which requires torch>=2.6 in current Transformers releases.
            for token_type_key in ("mm_token_type_ids", "token_type_ids"):
                prompt_model_inputs.pop(token_type_key, None)
            print(
                f"[Worker:{model_spec['key']}] Disabled Gemma token_type_ids "
                f"for torch {torch.__version__}; torch>=2.6 is required for that mask path."
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

    model_type = model.config.model_type
    print(
        f"[Worker:{model_spec['key']}] Ready on {device_name} "
        f"(model_type={model_type}, family={model_family})"
    )

    return {
        "model_spec": model_spec,
        "device": device,
        "processor": processor,
        "model": model,
        "model_family": model_family,
        "vision_grad_input_key": vision_grad_input_key,
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
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = target_batch["labels"][:, 1:]
    valid_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = token_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_mask = valid_mask.to(token_log_probs.dtype)
    avg_nll = -(token_log_probs * token_mask).sum() / token_mask.sum()
    return -avg_nll


def build_vision_grad_inputs(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    vision_grad_input_key = state["vision_grad_input_key"]
    # Different models expose differentiable image inputs under different keys.
    pixel_values_ref = vision_inputs[vision_grad_input_key].detach().requires_grad_(True)
    vision_inputs_ref = dict(vision_inputs)
    vision_inputs_ref[vision_grad_input_key] = pixel_values_ref
    return pixel_values_ref, vision_inputs_ref


def target_loss(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> torch.Tensor:
    if state["target_loss_mode"] == "standard_ce":
        target_batch = state["target_batches"][0]
        if not backward:
            with torch.no_grad():
                # `target_score` is negative NLL, so minimizing target loss is `-target_score`.
                return -target_score(state, target_batch, vision_inputs)

        pixel_values_ref, vision_inputs_ref = build_vision_grad_inputs(state, vision_inputs)
        loss = -target_score(state, target_batch, vision_inputs_ref)
        vision_grad_input_key = state["vision_grad_input_key"]
        grad = torch.autograd.grad(loss, pixel_values_ref)[0]
        vision_inputs[vision_grad_input_key].backward(grad)
        return loss.detach()

    if not backward:
        with torch.no_grad():
            detached_scores = torch.stack(
                [target_score(state, target_batch, vision_inputs) for target_batch in state["target_batches"]]
            )
            # Multi-reference mode uses log-mean-exp over target scores for a smooth aggregate objective.
            return -(
                torch.logsumexp(detached_scores, dim=0)
                - detached_scores.new_tensor(len(state["target_batches"])).log()
            )

    pixel_values_ref, vision_inputs_ref = build_vision_grad_inputs(state, vision_inputs)
    scores = torch.stack(
        [target_score(state, target_batch, vision_inputs_ref) for target_batch in state["target_batches"]]
    )
    aggregate_loss = -(torch.logsumexp(scores, dim=0) - scores.new_tensor(len(state["target_batches"])).log())
    vision_grad_input_key = state["vision_grad_input_key"]
    pixel_values_grad = torch.autograd.grad(aggregate_loss, pixel_values_ref)[0]
    vision_inputs[vision_grad_input_key].backward(pixel_values_grad)
    return aggregate_loss.detach()


def untargeted_reference_loss(
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    backward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_batch = state["untargeted_reference_batch"]
    if reference_batch is None:
        raise RuntimeError("Untargeted mode requires a clean reference generation before evaluation or attack.")

    if not backward:
        with torch.no_grad():
            metric_loss = -target_score(state, reference_batch, vision_inputs)
            # Reported metric is NLL on the clean reference, while optimization maximizes that NLL.
            return metric_loss, -metric_loss

    pixel_values_ref, vision_inputs_ref = build_vision_grad_inputs(state, vision_inputs)
    optimization_loss = target_score(state, reference_batch, vision_inputs_ref)
    metric_loss = -optimization_loss
    vision_grad_input_key = state["vision_grad_input_key"]
    grad = torch.autograd.grad(optimization_loss, pixel_values_ref)[0]
    vision_inputs[vision_grad_input_key].backward(grad)
    return metric_loss.detach(), optimization_loss.detach()


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
        use_cache=True,
    )


def evaluate_metric_loss(state: dict, image_gpu: torch.Tensor) -> float | None:
    if state["attack_mode"] != "targeted" and state["untargeted_reference_batch"] is None:
        return None

    vision_inputs = build_vision_inputs(state, image_gpu)
    metric_loss, _ = attack_loss(state, vision_inputs)
    return float(metric_loss.item())


def evaluate_image(state: dict, image_cpu: torch.Tensor) -> dict:
    image_gpu = image_cpu.to(state["device"], non_blocking=True).squeeze(0)
    with torch.no_grad():
        generation = generate_from_image(state, image_gpu)
        loss = evaluate_metric_loss(state, image_gpu)
    return {"loss": loss, "generation": generation}


def evaluate_image_loss_only(state: dict, image_cpu: torch.Tensor) -> dict:
    image_gpu = image_cpu.to(state["device"], non_blocking=True).squeeze(0)
    with torch.no_grad():
        loss = evaluate_metric_loss(state, image_gpu)
    if loss is None:
        raise RuntimeError("Loss-only evaluation requires a target or untargeted reference batch.")
    return {"loss": loss}


def attack_step(state: dict, image_cpu: torch.Tensor) -> dict:
    x_adv = image_cpu.to(state["device"], non_blocking=True).requires_grad_(True)
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
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    run_log_path = worker_config.get("run_log_path")
    if isinstance(run_log_path, str) and run_log_path:
        try:
            log_handle = open(run_log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            response_queue.put(
                {
                    "type": "error",
                    "key": model_spec["key"],
                    "message": traceback.format_exc(),
                }
            )
            return
        sys.stdout = _WorkerTee(original_stdout, log_handle)
        sys.stderr = _WorkerTee(original_stderr, log_handle)
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
                print(f"[Worker:{model_spec['key']}] Starting evaluate", flush=True)
                started_at = time.perf_counter()
                result = evaluate_image(state, message["image"])
                response_queue.put({"type": "evaluate", "key": model_spec["key"], **result})
                elapsed = time.perf_counter() - started_at
                print(
                    f"[Worker:{model_spec['key']}] Finished evaluate in {elapsed:.1f}s",
                    flush=True,
                )
                continue
            if command == "evaluate_loss_only":
                result = evaluate_image_loss_only(state, message["image"])
                response_queue.put({"type": "evaluate_loss_only", "key": model_spec["key"], **result})
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
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.close()


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
    image_cpu = image_cpu.detach()
    if image_cpu.device.type != "cpu":
        image_cpu = image_cpu.cpu()
    if not image_cpu.is_contiguous():
        image_cpu = image_cpu.contiguous()
    for worker in workers:
        worker["request_queue"].put({"command": command, "image": image_cpu})

    total_workers = len(workers)
    pending_workers = {
        worker["model_spec"]["key"]: worker
        for worker in workers
    }
    messages_by_key = {}
    started_at = time.perf_counter()
    last_status_at = started_at

    while pending_workers:
        progress_made = False
        for key, worker in list(pending_workers.items()):
            try:
                # Poll non-blocking so one slow worker does not stall checks for the others.
                message = worker["response_queue"].get_nowait()
            except queue.Empty:
                if not worker["process"].is_alive():
                    # Surface dead workers immediately instead of waiting for a queue timeout.
                    raise RuntimeError(
                        f"Worker {key} exited before sending a {expected_type!r} response."
                    )
                continue

            progress_made = True
            if message["type"] == "error":
                raise RuntimeError(f"Worker {key} failed:\n{message['message']}")
            if message["type"] != expected_type:
                raise RuntimeError(
                    f"Expected {expected_type} result from {key}, got {message['type']!r}."
                )

            messages_by_key[key] = message
            pending_workers.pop(key)
            if command != "attack_step":
                print(
                    f"[Info] Received {command} result from {key} "
                    f"({len(messages_by_key)}/{total_workers})",
                    flush=True,
                )

        now = time.perf_counter()
        if command != "attack_step" and pending_workers and now - last_status_at >= 10.0:
            pending_list = ", ".join(sorted(pending_workers))
            print(
                f"[Info] Waiting for {command} results from: {pending_list} "
                f"({now - started_at:.1f}s elapsed)",
                flush=True,
            )
            last_status_at = now

        if not progress_made:
            time.sleep(0.1)

    return [messages_by_key[worker["model_spec"]["key"]] for worker in workers]


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


def evaluate_workers_loss_only(workers: list[dict], image_cpu: torch.Tensor) -> dict[str, float]:
    return {
        message["key"]: message["loss"]
        for message in dispatch_worker_command(
            workers,
            "evaluate_loss_only",
            image_cpu,
            expected_type="evaluate_loss_only",
        )
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
