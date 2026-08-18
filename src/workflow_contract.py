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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    },
    "hunyuan_ocr": {
        "model_name": "tencent/HunyuanOCR",
        "model_family": "hunyuan",
        "ocr_prompt": "提取图中的文字。",
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
        "prompt": "Please repeat the text shown in the image.",
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
        config["prompt"] = OCR_MODELS["donut"]["ocr_prompt"]
        return config

    config = {
        "schema_version": SCHEMA_VERSION,
        "name": f"example-{workflow}",
        "workflow": workflow,
        "inputs": {"image": "data/images/6.png"},
        "models": {
            "attack": "llava_1_5_7b_hf",
            "transfer": ["instructblip_vicuna_7b"],
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


def _model_keys(config: dict[str, Any]) -> list[str]:
    raw = config.get("models")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        keys: list[str] = []
        for field in ("attack", "transfer", "models", "inference"):
            value = raw.get(field)
            if isinstance(value, str):
                keys.append(value)
            elif isinstance(value, list):
                keys.extend(value)
        return keys
    return []


def _model_table(workflow: str) -> dict[str, dict[str, Any]]:
    if workflow.startswith("ocr"):
        return OCR_MODELS
    if workflow == "vlm_inference":
        return VLM_INFERENCE_MODELS
    return VLM_ATTACK_MODELS


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

    model_table = _model_table(workflow)
    keys = _model_keys(resolved)
    if not keys:
        raise ValueError("models must select at least one model key.")
    unknown = [key for key in keys if not isinstance(key, str) or key not in model_table]
    if unknown:
        unknown_display = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown model key(s): {unknown_display}.")
    resolved["model_revisions"] = [model_table[key]["model_name"] for key in keys]

    if workflow.startswith("ocr"):
        selected = keys[0]
        model_defaults = OCR_MODELS[selected]
        if resolved.get("prompt") is None:
            resolved["prompt"] = model_defaults.get("ocr_prompt")
        resolved["ocr_model_defaults"] = {
            key: value for key, value in model_defaults.items() if key in {"base_size", "image_size", "crop_mode"}
        }
    else:
        resolved["devices"] = _resolve_devices(resolved, keys)
        if "attack" in resolved and resolved["attack"]["mode"] == "targeted" and not resolved["attack"].get("target_texts"):
            raise ValueError("attack.target_texts must be non-empty for a targeted attack.")
    _validate_values(resolved)
    return resolved


def _resolve_devices(config: dict[str, Any], keys: list[str]) -> list[str]:
    explicit = config.get("devices")
    if explicit is None:
        base = str(config.get("device", "cuda:0"))
        try:
            start = int(base.split(":", 1)[1]) if ":" in base else 0
        except ValueError as exc:
            raise ValueError("device must look like cuda:0.") from exc
        return [f"cuda:{start + index}" for index in range(len(keys))]
    if not isinstance(explicit, list) or len(explicit) != len(keys):
        raise ValueError("devices must be a list with one explicit CUDA device per selected model.")
    return [str(device) for device in explicit]


def _validate_values(config: dict[str, Any]) -> None:
    device = config.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("device must be an explicit CUDA device such as 'cuda:0'.")
    generation = config.get("generation", {})
    if not isinstance(generation.get("max_new_tokens"), int) or generation["max_new_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be a positive integer.")
    if "seed" not in config or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")
    if "attack" in config:
        attack = config["attack"]
        if attack.get("epsilon", 0) <= 0 or attack["epsilon"] > 1:
            raise ValueError("attack.epsilon must be in (0, 1].")
        if attack.get("alpha", 0) <= 0 or attack["alpha"] > attack["epsilon"]:
            raise ValueError("attack.alpha must be in (0, attack.epsilon].")
        if not isinstance(attack.get("steps"), int) or attack["steps"] <= 0:
            raise ValueError("attack.steps must be a positive integer.")
    if config["workflow"].startswith("vlm"):
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
        if generation.get("temperature", 0) <= 0:
            raise ValueError("generation.temperature must be positive.")
        if not 0 < generation.get("top_p", 0) <= 1:
            raise ValueError("generation.top_p must be in (0, 1].")
        eot = config.get("eot", {})
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
        devices = config.get("devices", [])
        if len(set(devices)) != len(devices):
            raise ValueError("Each VLM worker must use a distinct CUDA device.")
        if any(not re.fullmatch(r"cuda:\d+", device_name) for device_name in devices):
            raise ValueError("Every VLM worker device must be explicit, such as 'cuda:0'.")


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
        "sys_executable": sys.executable,
        "conda_environment": conda_env,
        "python_version": sys.version.split()[0],
        "library_versions": package_versions(),
        "cuda_devices": cuda_devices,
        "requested_model_revisions": config.get("model_revisions", []),
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
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
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
        for key in sorted(metrics):
            value = metrics[key]
            lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True, ensure_ascii=False)}")
        lines.append("")
    if results.get("raw_outputs"):
        lines.extend(["## Outputs", "", "Raw generations/transcriptions are preserved in `results.json`.", ""])
    return "\n".join(lines)


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
    requested = [config.get("device", "cuda:0")]
    requested.extend(config.get("devices", []))
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
