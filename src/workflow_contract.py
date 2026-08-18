"""Environment-neutral manifests, defaults, and run-artifact helpers."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
WORKFLOWS = (
    "ocr_attack",
    "ocr_inference",
    "ocr_pipeline",
    "vlm_attack",
    "vlm_inference",
    "vlm_pipeline",
)
OCR_INTERPRETER = "/home/jmadden2/anaconda3/envs/ocr/bin/python"
VLM_INTERPRETER = "/home/jmadden2/anaconda3/envs/llm-misuse/bin/python"

OCR_MODELS = {
    "imgscope": {
        "model_name": "prithivMLmods/Imgscope-OCR-2B-0527",
        "model_family": "qwen",
        "ocr_prompt": "Read all text in the image and output only the extracted text.",
    },
    "deepseek_ocr_2": {
        "model_name": "deepseek-ai/DeepSeek-OCR-2",
        "model_family": "deepseek",
        "ocr_prompt": "<image>\nExtract all text.",
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,
    },
    "qianfan_ocr": {
        "model_name": "baidu/Qianfan-OCR",
        "model_family": "qianfan",
        "ocr_prompt": "Read all text in the image and output only the extracted text.",
        "available": False,
        "unavailable_reason": "The canonical OCR environment does not provide Qianfan OCR's required Transformers model class.",
    },
    "hunyuan_ocr": {
        "model_name": "tencent/HunyuanOCR",
        "model_family": "hunyuan",
        "ocr_prompt": "提取图中的文字。",
        "available": False,
        "unavailable_reason": "The canonical OCR environment does not provide HunyuanOCR's required Transformers model class.",
    },
    "donut": {
        "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
        "model_family": "encoder_decoder_donut",
        "ocr_prompt": "<s_docvqa><s_question>Read all text in the image and return what is says.</s_question><s_answer>",
    },
    "nougat": {
        "model_name": "facebook/nougat-small",
        "model_family": "encoder_decoder_nougat",
        "ocr_prompt": None,
    },
}

VLM_ATTACK_MODELS = {
    "aya_vision_8b": {
        "model_name": "CohereLabs/aya-vision-8b",
        "model_family": "aya_vision",
        "group": "siglip2",
    },
    "jina_vlm": {
        "model_name": "jinaai/jina-vlm",
        "model_family": "jina_vlm",
        "group": "siglip2",
        "auto_model_class": "causal_lm",
        "trust_remote_code": True,
    },
    "lfm2_5_vl_1_6b": {
        "model_name": "LiquidAI/LFM2.5-VL-1.6B",
        "model_family": "lfm2_vl",
        "group": "siglip2",
    },
    "gemma3_4b_it": {
        "model_name": "google/gemma-3-4b-it",
        "model_family": "auto",
        "group": "siglip",
    },
    "granite_vision_3_2": {
        "model_name": "ibm-granite/granite-vision-3.2-2b",
        "model_family": "auto",
        "group": "siglip",
    },
    "smolvlm_2b_instruct": {
        "model_name": "HuggingFaceTB/SmolVLM-Instruct",
        "model_family": "auto",
        "group": "siglip",
    },
    "llava_1_5_7b_hf": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "model_family": "auto",
        "group": "clip",
    },
    "instructblip_vicuna_7b": {
        "model_name": "Salesforce/instructblip-vicuna-7b",
        "model_family": "instructblip",
        "group": "clip",
    },
}

VLM_INFERENCE_MODELS = {
    "paligemma": {"model_name": "google/paligemma2-3b-mix-448", "family": "paligemma"},
    "llava_onevision": {
        "model_name": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "family": "llava_onevision",
    },
    "idefics2": {"model_name": "HuggingFaceM4/idefics2-8b", "family": "idefics2"},
    "smolvlm": {"model_name": "HuggingFaceTB/SmolVLM-Instruct", "family": "smolvlm"},
    "internvl3_1b_hf": {"model_name": "OpenGVLab/InternVL3-1B-hf", "family": "internvl"},
    "openflamingo_4b": {
        "model_name": "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
        "family": "openflamingo",
    },
    "molmo_7b_d_0924": {"model_name": "allenai/Molmo-7B-D-0924", "family": "molmo"},
}

VLM_ATTACK_GROUPS = {
    "siglip2": ["aya_vision_8b", "jina_vlm", "lfm2_5_vl_1_6b"],
    "siglip": ["gemma3_4b_it", "granite_vision_3_2", "smolvlm_2b_instruct"],
    "clip": ["llava_1_5_7b_hf", "instructblip_vicuna_7b"],
}


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


REPO_ROOT = find_repo_root()


def _ocr_defaults(workflow: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "device": "cuda:0",
        "seed": 0,
        "generation": {"max_new_tokens": 128},
        "prompt": None,
    }
    if workflow in {"ocr_attack", "ocr_pipeline"}:
        defaults.update(
            {
                "attack": {
                    "epsilon": 12 / 255,
                    "alpha": 2 / 255,
                    "steps": 500,
                    "random_start": False,
                }
            }
        )
    return defaults


def _vlm_defaults(workflow: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "device": "cuda:0",
        "seed": 0,
        "prompt": (
            "Solve the problems shown in the image. Show steps and give the final answers."
            if workflow in {"vlm_attack", "vlm_pipeline"}
            else "Please repeat the text shown in the image."
        ),
        "generation": {
            "max_new_tokens": 128,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.8,
        },
        "eot": {
            "enabled": False,
            "train_samples": 4,
            "eval_samples": 1,
            "rotation_degrees": 0.0,
            "perspective_distortion": 0.0,
            "crop_scale": [1.0, 1.0],
            "crop_ratio": [1.0, 1.0],
            "color_jitter_brightness": 0.1,
            "color_jitter_contrast": 0.1,
            "color_jitter_saturation": 0.1,
            "gaussian_noise_std": 0.02,
        },
    }
    if workflow in {"vlm_attack", "vlm_pipeline"}:
        defaults.update(
            {
                "attack": {
                    "epsilon": 64 / 255,
                    "alpha": 4 / 1000,
                    "steps": 1500,
                    "mode": "untargeted",
                    "target_loss_mode": "standard_ce",
                    "target_texts": [
                        "I cannot help with that",
                        "No",
                        "I'm sorry, I cant answer that.",
                    ],
                    "cross_model_optimization_mode": "mean_ce",
                    "softminimax_temperature": 1.0,
                }
            }
        )
    if workflow == "vlm_pipeline":
        defaults["transfer_device"] = defaults["device"]
    return defaults


def example_manifest(workflow: str) -> dict[str, Any]:
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r}.")
    if workflow.startswith("ocr"):
        config = {
            "schema_version": SCHEMA_VERSION,
            "name": f"example-{workflow}",
            "workflow": workflow,
            "inputs": {"image": "data/images/15.png"},
            "models": {
                "attack": "donut",
                "transfer": ["deepseek_ocr_2", "imgscope", "nougat"],
            }
            if workflow == "ocr_pipeline"
            else (["deepseek_ocr_2", "imgscope", "donut", "nougat"] if workflow == "ocr_inference" else "donut"),
        }
        if workflow == "ocr_inference":
            config["inputs"] = {"image": "results/nougat_ocr_12_adv_15.png"}
        config.update(_ocr_defaults(workflow))
        if workflow in {"ocr_attack", "ocr_pipeline"}:
            config["prompt"] = OCR_MODELS["donut"]["ocr_prompt"]
        return config

    config = {
        "schema_version": SCHEMA_VERSION,
        "name": f"example-{workflow}",
        "workflow": workflow,
        "inputs": {"image": "data/images/6.png"},
        "models": {
            "attack": "llava_1_5_7b_hf",
            "transfer": ["internvl3_1b_hf"],
        }
        if workflow == "vlm_pipeline"
        else (["internvl3_1b_hf", "openflamingo_4b", "molmo_7b_d_0924"] if workflow == "vlm_inference" else "llava_1_5_7b_hf"),
    }
    if workflow == "vlm_inference":
        config["inputs"] = {"image": "results/llava_textgen_multi_gpu_adv_6_clip_64.png"}
    config.update(_vlm_defaults(workflow))
    return config


def _deep_merge(defaults: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _path_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _as_model_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{label} must be a model key or a non-empty list of model keys.")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{label} must be a model key or a non-empty list of model keys.")
    return list(values)


def _selected_model_keys(workflow: str, models: Any) -> tuple[list[str], list[str]]:
    """Return attack and sequential-inference selections after checking shape."""
    if workflow == "ocr_attack":
        if not isinstance(models, str) or not models.strip():
            raise ValueError("ocr_attack models must be one OCR model key.")
        return [models], []
    if workflow == "ocr_inference":
        return [], _as_model_list(models, "ocr_inference models")
    if workflow == "ocr_pipeline":
        if not isinstance(models, dict) or set(models) - {"attack", "transfer"}:
            raise ValueError("ocr_pipeline models must contain only attack and transfer fields.")
        attack = models.get("attack")
        if not isinstance(attack, str) or not attack.strip():
            raise ValueError("ocr_pipeline models.attack must be one OCR model key.")
        return [attack], _as_model_list(models.get("transfer"), "ocr_pipeline models.transfer")
    if workflow == "vlm_attack":
        if isinstance(models, dict):
            raise ValueError("vlm_attack models must be one or more VLM attack model keys, not a pipeline object.")
        return _as_model_list(models, "vlm_attack models"), []
    if workflow == "vlm_inference":
        if isinstance(models, dict):
            raise ValueError("vlm_inference models must be one or more VLM inference model keys, not a pipeline object.")
        return [], _as_model_list(models, "vlm_inference models")
    if workflow == "vlm_pipeline":
        if not isinstance(models, dict) or set(models) - {"attack", "transfer"}:
            raise ValueError("vlm_pipeline models must contain only attack and transfer fields.")
        return _as_model_list(models.get("attack"), "vlm_pipeline models.attack"), _as_model_list(
            models.get("transfer"), "vlm_pipeline models.transfer"
        )
    raise ValueError(f"Unsupported workflow {workflow!r}.")


def resolve_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Manifest must be a JSON object.")
    workflow = raw.get("workflow")
    if workflow not in WORKFLOWS:
        supported = ", ".join(WORKFLOWS)
        raise ValueError(f"workflow must be one of: {supported}; got {workflow!r}.")
    defaults = _ocr_defaults(workflow) if workflow.startswith("ocr") else _vlm_defaults(workflow)
    resolved = _deep_merge(defaults, raw)
    if resolved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    if not isinstance(resolved.get("name"), str) or not resolved["name"].strip():
        raise ValueError("name must be a non-empty string.")
    inputs = resolved.get("inputs")
    if not isinstance(inputs, dict) or "image" not in inputs:
        raise ValueError("inputs.image is required.")
    resolved["inputs"] = dict(inputs)
    resolved["inputs"]["image"] = _path_value(inputs["image"], "inputs.image")
    if "clean" in inputs:
        resolved["inputs"]["clean"] = _path_value(inputs["clean"], "inputs.clean")
    if "adversarial" in inputs:
        resolved["inputs"]["adversarial"] = _path_value(inputs["adversarial"], "inputs.adversarial")

    attack_keys, transfer_keys = _selected_model_keys(workflow, resolved.get("models"))
    keys = attack_keys + transfer_keys
    attack_table = OCR_MODELS if workflow.startswith("ocr") else VLM_ATTACK_MODELS
    transfer_table = OCR_MODELS if workflow.startswith("ocr") else VLM_INFERENCE_MODELS
    unknown_attack = [key for key in attack_keys if key not in attack_table]
    unknown_transfer = [key for key in transfer_keys if key not in transfer_table]
    if unknown_attack or unknown_transfer:
        parts = []
        if unknown_attack:
            parts.append("attack=" + ", ".join(sorted(map(str, unknown_attack))))
        if unknown_transfer:
            parts.append("transfer=" + ", ".join(sorted(map(str, unknown_transfer))))
        raise ValueError("Unknown model key(s): " + "; ".join(parts) + ".")
    unavailable = [
        key for key in keys
        if not (attack_table if key in attack_keys else transfer_table)[key].get("available", True)
    ]
    if unavailable:
        details = "; ".join(
            f"{key}: {(attack_table if key in attack_keys else transfer_table)[key].get('unavailable_reason', 'not available')}"
            for key in unavailable
        )
        raise ValueError(f"Selected model(s) are unavailable in the canonical environment: {details}")
    resolved["model_ids"] = {
        key: (attack_table if key in attack_keys else transfer_table)[key]["model_name"]
        for key in dict.fromkeys(keys)
    }
    # No model revision resolver is implemented; never mislabel model IDs as revisions.
    requested_revisions = raw.get("revisions", raw.get("model_revisions"))
    if requested_revisions is None:
        requested_revisions = {}
    if not isinstance(requested_revisions, dict):
        raise ValueError("revisions must be an object mapping selected model keys to revision identifiers.")
    if any(key not in keys or not isinstance(value, str) or not value.strip() for key, value in requested_revisions.items()):
        raise ValueError("revisions keys must be selected model keys and values must be non-empty strings.")
    resolved["revisions"] = dict(requested_revisions)
    # Keep the compatibility field honest: it contains only explicit revisions,
    # never model IDs inferred from the catalog.
    resolved["model_revisions"] = dict(requested_revisions)

    if workflow.startswith("ocr"):
        if workflow in {"ocr_attack", "ocr_pipeline"}:
            model_defaults = OCR_MODELS[attack_keys[0]]
            if resolved.get("prompt") is None:
                resolved["prompt"] = model_defaults.get("ocr_prompt")
            resolved["ocr_model_defaults"] = {
                key: value for key, value in model_defaults.items() if key in {"base_size", "image_size", "crop_mode"}
            }
        if "inference_prompts" in resolved:
            prompts = resolved["inference_prompts"]
            if not isinstance(prompts, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in prompts.items()
            ):
                raise ValueError("inference_prompts must map model keys to prompt strings.")
            invalid_prompt_keys = sorted(set(prompts) - set(transfer_keys or attack_keys))
            if invalid_prompt_keys:
                raise ValueError("inference_prompts contains model keys not selected for inference: " + ", ".join(invalid_prompt_keys))
    else:
        if workflow == "vlm_inference" and "devices" in resolved:
            raise ValueError("vlm_inference uses device only; remove devices.")
        if workflow == "vlm_pipeline":
            resolved["transfer_device"] = (
                resolved.get("transfer_device") if "transfer_device" in raw else resolved.get("device", "cuda:0")
            )
        if workflow in {"vlm_attack", "vlm_pipeline"}:
            resolved["devices"] = _resolve_devices(resolved, attack_keys)
        if (
            isinstance(resolved.get("attack"), dict)
            and resolved["attack"].get("mode") == "targeted"
            and not resolved["attack"].get("target_texts")
        ):
            raise ValueError("attack.target_texts must be non-empty for a targeted attack.")
    if workflow.startswith("ocr") and "devices" in raw:
        raise ValueError("OCR workflows use device only; remove devices.")
    if workflow != "vlm_pipeline" and "transfer_device" in raw:
        raise ValueError("transfer_device is supported only by vlm_pipeline.")
    _validate_values(resolved)
    return resolved


def _resolve_devices(config: dict[str, Any], keys: list[str]) -> list[str]:
    explicit = config.get("devices")
    if explicit is None:
        if len(keys) != 1:
            raise ValueError("devices must explicitly list one unique CUDA device per selected VLM attack model.")
        return [str(config.get("device", "cuda:0"))]
    if not isinstance(explicit, list) or len(explicit) != len(keys):
        raise ValueError("devices must be a list with one explicit CUDA device per selected model.")
    return [str(device) for device in explicit]


def _validate_values(config: dict[str, Any]) -> None:
    device = config.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("device must be an explicit CUDA device such as 'cuda:0'.")
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object.")
    if not isinstance(generation.get("max_new_tokens"), int) or isinstance(generation["max_new_tokens"], bool) or generation["max_new_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be a positive integer.")
    if "seed" not in config or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")
    if "attack" in config:
        attack = config["attack"]
        if not isinstance(attack, dict):
            raise ValueError("attack must be an object.")
        epsilon = attack.get("epsilon")
        alpha = attack.get("alpha")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0 or epsilon > 1:
            raise ValueError("attack.epsilon must be in (0, 1].")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0 or alpha > epsilon:
            raise ValueError("attack.alpha must be in (0, attack.epsilon].")
        if not isinstance(attack.get("steps"), int) or attack["steps"] <= 0:
            raise ValueError("attack.steps must be a positive integer.")
    if config["workflow"].startswith("vlm"):
        if not isinstance(config.get("prompt"), str) or not config["prompt"].strip():
            raise ValueError("prompt must be a non-empty string for VLM workflows.")
        attack = config.get("attack", {})
        if attack and attack.get("mode") not in {"targeted", "untargeted"}:
            raise ValueError("attack.mode must be 'targeted' or 'untargeted'.")
        if attack and attack.get("target_loss_mode") not in {"standard_ce", "multi_reference"}:
            raise ValueError("attack.target_loss_mode must be 'standard_ce' or 'multi_reference'.")
        if attack and attack.get("cross_model_optimization_mode") not in {"mean_ce", "softminimax"}:
            raise ValueError("attack.cross_model_optimization_mode must be 'mean_ce' or 'softminimax'.")
        if attack and attack.get("softminimax_temperature", 0) <= 0:
            raise ValueError("attack.softminimax_temperature must be positive.")
        if attack and attack.get("target_texts") is not None:
            target_texts = attack["target_texts"]
            if not isinstance(target_texts, list) or any(not isinstance(value, str) or not value for value in target_texts):
                raise ValueError("attack.target_texts must be a list of non-empty strings.")
        temperature = generation.get("temperature")
        top_p = generation.get("top_p")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
            raise ValueError("generation.temperature must be positive.")
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0 < top_p <= 1:
            raise ValueError("generation.top_p must be in (0, 1].")
        eot = config.get("eot", {})
        if not isinstance(eot, dict):
            raise ValueError("eot must be an object.")
        for field in ("train_samples", "eval_samples"):
            if not isinstance(eot.get(field), int) or eot[field] <= 0:
                raise ValueError(f"eot.{field} must be a positive integer.")
        for field in ("rotation_degrees", "perspective_distortion", "gaussian_noise_std", "color_jitter_brightness", "color_jitter_contrast", "color_jitter_saturation"):
            if not isinstance(eot.get(field), (int, float)) or eot[field] < 0:
                raise ValueError(f"eot.{field} must be non-negative.")
        if eot.get("perspective_distortion", 0) > 1:
            raise ValueError("eot.perspective_distortion must be at most 1.")
        for field in ("crop_scale", "crop_ratio"):
            value = eot.get(field)
            if not isinstance(value, list) or len(value) != 2 or value[0] <= 0 or value[1] < value[0]:
                raise ValueError(f"eot.{field} must be a positive [low, high] pair.")
        if config["workflow"] in {"vlm_attack", "vlm_pipeline"}:
            devices = config.get("devices", [])
            if len(set(devices)) != len(devices):
                raise ValueError("Each VLM attack worker must use a distinct CUDA device.")
            if any(not isinstance(device_name, str) or not re.fullmatch(r"cuda:\d+", device_name) for device_name in devices):
                raise ValueError("Every VLM attack worker device must be explicit, such as 'cuda:0'.")
        if config["workflow"] == "vlm_pipeline":
            transfer_device = config.get("transfer_device")
            if not isinstance(transfer_device, str) or not re.fullmatch(r"cuda:\d+", transfer_device):
                raise ValueError("transfer_device must be an explicit CUDA device such as 'cuda:0'.")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return result[:80] or "run"


def create_run_dir(config: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = REPO_ROOT / "results" / "runs" / f"{timestamp}-{slug(config['name'])}-{config_hash(config)}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    (run_dir / "artifacts").mkdir()
    return run_dir


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "torchvision", "Pillow", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def runtime_metadata(config: dict[str, Any]) -> dict[str, Any]:
    prefix_name = Path(sys.prefix).name
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not conda_env or (conda_env == "base" and prefix_name != "base"):
        conda_env = prefix_name
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all visible devices")
    return {
        "git_commit": _git_commit(),
        **_git_worktree_metadata(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "sys_executable": sys.executable,
        "conda_environment": conda_env,
        "python_version": sys.version.split()[0],
        "library_versions": package_versions(),
        "cuda_devices": cuda_devices,
        "model_ids": config.get("model_ids", {}),
        "requested_revisions": config.get("revisions", {}),
        "requested_model_revisions": config.get("model_revisions", {}),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_metadata() -> dict[str, Any]:
    """Capture enough local state to distinguish dirty runs from HEAD-only runs."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src", "AGENTS.md", "plans"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_dirty": None, "git_diff_hash": None, "git_changed_files": []}
    changed_files = []
    for line in status.splitlines():
        if len(line) >= 4:
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            changed_files.append(path)
    payload = status + "\n" + diff + "\n" + "\n".join(sorted(changed_files))
    return {
        "git_dirty": bool(status.strip()),
        "git_diff_hash": hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest(),
        "git_changed_files": sorted(set(changed_files)),
    }


class EventLog:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.log_path = run_dir / "run.log"

    def __call__(self, event: str, **payload: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
        line = f"[{record['timestamp']}] {event}"
        if payload:
            line += " " + canonical_json(payload)
        print(line, flush=True)


def write_status(run_dir: Path, status: dict[str, Any]) -> None:
    (run_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def relative_artifact_paths(value: Any, run_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: relative_artifact_paths(item, run_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_artifact_paths(item, run_dir) for item in value]
    if isinstance(value, str):
        path = Path(value)
        try:
            return str(path.resolve().relative_to(run_dir.resolve()))
        except ValueError:
            return value
    return value


def deterministic_summary(config: dict[str, Any], results: dict[str, Any], status: str) -> str:
    lines = [
        f"# {config['name']}",
        "",
        f"- Workflow: `{config['workflow']}`",
        f"- Status: `{status}`",
        "",
        "This summary reports behavioral differences between saved clean and adversarial outputs where applicable; textual changes are not semantic correctness.",
        "",
    ]
    metrics = results.get("metrics", {})
    if metrics:
        lines.extend(["## Metrics", ""])
        scalar_metrics = _flatten_scalar_metrics(metrics)
        for key in sorted(scalar_metrics):
            lines.append(f"- `{key}`: {json.dumps(scalar_metrics[key], sort_keys=True, ensure_ascii=False)}")
        complex_metrics = sorted(set(metrics) - {key.split(".", 1)[0] for key in scalar_metrics})
        if complex_metrics:
            lines.append("")
            lines.append("Nested/non-scalar metric objects are preserved in `results.json`.")
        lines.append("")
    if results.get("raw_outputs"):
        lines.extend(["## Outputs", "", "Raw generations/transcriptions are preserved in `results.json`.", ""])
    return "\n".join(lines)


def _flatten_scalar_metrics(value: Any, prefix: str = "") -> dict[str, int | float | bool]:
    """Flatten only numeric and boolean leaves for conservative comparisons."""
    if isinstance(value, bool):
        return {prefix: value} if prefix else {}
    if isinstance(value, (int, float)) and not isinstance(value, complex):
        return {prefix: value} if prefix else {}
    if isinstance(value, dict):
        flattened: dict[str, int | float | bool] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalar_metrics(value[key], child_prefix))
        return flattened
    return {}


def expected_interpreter(workflow: str) -> str:
    return OCR_INTERPRETER if workflow.startswith("ocr") else VLM_INTERPRETER


def check_interpreter(workflow: str, command_hint: str) -> None:
    expected_command = expected_interpreter(workflow)
    expected = Path(expected_command).resolve()
    actual = Path(sys.executable).resolve()
    expected_prefix = expected.parent.parent
    actual_prefix = Path(sys.prefix).resolve()
    if actual != expected or actual_prefix != expected_prefix:
        raise RuntimeError(
            f"Workflow {workflow!r} requires {expected_command} with prefix {expected_prefix}. "
            f"Current interpreter is {actual} with prefix {actual_prefix}. "
            f"Run the canonical command with the correct interpreter: {expected_command} {command_hint}"
        )


def check_cuda_devices(config: dict[str, Any]) -> None:
    """Check CUDA only after the workflow environment has been selected."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This workflow requires CUDA, but CUDA is not available in the selected environment.")
    workflow = config.get("workflow")
    if workflow == "vlm_attack":
        # With an explicit worker list, the top-level device is not used.
        requested = list(config.get("devices", [config.get("device", "cuda:0")]))
    elif workflow == "vlm_pipeline":
        requested = list(config.get("devices", [config.get("device", "cuda:0")]))
        requested.append(config.get("transfer_device", config.get("device", "cuda:0")))
    else:
        requested = [config.get("device", "cuda:0")]
    count = torch.cuda.device_count()
    for device_name in requested:
        index = int(device_name.split(":", 1)[1])
        if index >= count:
            raise RuntimeError(f"Configured device {device_name} is not visible; {count} CUDA device(s) are available.")


def batch_entries(raw: dict[str, Any], batch_path: Path) -> list[dict[str, Any]]:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"batch schema_version must be {SCHEMA_VERSION}.")
    entries = raw.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("batch.runs must be a non-empty list.")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), str):
            raise ValueError(f"batch.runs[{index}] must contain a string config path.")
        config_path = Path(entry["config"])
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        item = {"config": str(config_path.resolve())}
        if "set" in entry:
            if not isinstance(entry["set"], dict):
                raise ValueError(f"batch.runs[{index}].set must be an object.")
            item["set"] = entry["set"]
        result.append(item)
    return result
