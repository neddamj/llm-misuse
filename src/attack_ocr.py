import gc
import importlib.util
import math
import os
import traceback
from pathlib import Path
from types import MethodType
from typing import Any
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    DonutProcessor,
    NougatProcessor,
    Qwen2VLForConditionalGeneration,
    VisionEncoderDecoderModel,
)
from transformers.utils import logging as transformers_logging

from attacks.common import find_repo_root, load_image_tensor, project_delta, save_image_tensor
from attacks.prompting import build_chat_prompt_inputs, build_teacher_forced_batch, generate_greedy_text
from attacks.vision import build_qwen_vision_inputs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()

MODEL_CONFIGS = {
    "imgscope": {
        "model_name": "prithivMLmods/Imgscope-OCR-2B-0527",
        "model_family": "qwen",
        "ocr_prompt": "Read all text in the image and output only the extracted text.",
        "result_prefix": "imgscope_ocr_pgd",
    },
    "deepseek_ocr_2": {
        "model_name": "deepseek-ai/DeepSeek-OCR-2",
        "model_family": "deepseek",
        "ocr_prompt": "<image>\nExtract all text.",
        "result_prefix": "deepseek_ocr_2_pgd",
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,
    },
    "qianfan_ocr": {
        "model_name": "baidu/Qianfan-OCR",
        "model_family": "qianfan",
        "ocr_prompt": "Read all text in the image and output only the extracted text.",
        "result_prefix": "qianfan_ocr_pgd",
    },
    "hunyuan_ocr": {
        "model_name": "tencent/HunyuanOCR",
        "model_family": "hunyuan",
        "ocr_prompt": "提取图中的文字。",
        "result_prefix": "hunyuan_ocr_pgd",
    },
    "donut": {
        "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
        "model_family": "encoder_decoder_donut",
        "ocr_prompt": "<s_docvqa><s_question>Read all text in the image and return what is says.</s_question><s_answer>",
        "result_prefix": "donut_ocr_pgd",
    },
    "nougat": {
        "model_name": "facebook/nougat-small",
        "model_family": "encoder_decoder_nougat",
        "ocr_prompt": None,
        "result_prefix": "nougat_ocr_pgd",
    },
}

NUM, DEN = 12, 255
EPSILON = NUM / DEN
ALPHA = 2 / 255
STEPS = 500
MAX_NEW_TOKENS = 128
RANDOM_START = False

MODEL_INPUT_SIZE = 448
DEEPSEEK_IMAGE_TOKEN_ID = 128815
DEEPSEEK_PATCH_SIZE = 16
DEEPSEEK_DOWNSAMPLE_RATIO = 5
DEEPSEEK_NORMALIZE_MEAN = 0.5
DEEPSEEK_NORMALIZE_STD = 0.5
ENCODER_DECODER_FAMILIES = {"encoder_decoder_donut", "encoder_decoder_nougat"}

REPO_ROOT = find_repo_root(Path(__file__).resolve())
RESULTS_DIR = REPO_ROOT / "results"

MODEL_KEY = "donut"  # Options: "deepseek_ocr_2", "imgscope", "donut", "nougat"
IMG_IDX = 15  # Options: 0-15, corresponds to the image index in data/images/{IMG_IDX}.png
SOURCE_IMAGE_PATH = REPO_ROOT / "data" / "images" / f"{IMG_IDX}.png"
OUTPUT_ADV_PATH = RESULTS_DIR / f"{MODEL_KEY}_ocr_{NUM}_adv_{IMG_IDX}.png"


def is_encoder_decoder_family(model_family: str) -> bool:
    return model_family in ENCODER_DECODER_FAMILIES


def build_run_config() -> dict:
    if MODEL_KEY not in MODEL_CONFIGS:
        supported_keys = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"MODEL_KEY must be one of: {supported_keys}. Got {MODEL_KEY!r}.")

    model_config = MODEL_CONFIGS[MODEL_KEY]
    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE_PATH}")

    run_config = {
        "img_idx": IMG_IDX,
        "model_key": MODEL_KEY,
        "model_name": model_config["model_name"],
        "model_family": model_config["model_family"],
        "ocr_prompt": model_config["ocr_prompt"],
        "source_image_path": SOURCE_IMAGE_PATH,
        "output_adv_path": OUTPUT_ADV_PATH,
        "output_report_path": OUTPUT_ADV_PATH.with_name(f"{OUTPUT_ADV_PATH.stem}_report.txt"),
    }
    for optional_key in ("base_size", "image_size", "crop_mode"):
        if optional_key in model_config:
            run_config[optional_key] = model_config[optional_key]
    return run_config


def ensure_qianfan_transformers_support() -> None:
    required_symbols = ("QianfanOCRProcessor", "QianfanOCRForConditionalGeneration")
    missing_symbols = [symbol for symbol in required_symbols if not hasattr(transformers, symbol)]
    if missing_symbols:
        missing_display = ", ".join(missing_symbols)
        raise RuntimeError(
            "Qianfan-OCR support requires a newer Transformers build with native QianfanOCR support. "
            f"Missing symbols: {missing_display}. Current transformers version: {transformers.__version__}."
        )


def ensure_hunyuan_transformers_support() -> None:
    required_symbols = ("HunYuanVLProcessor", "HunYuanVLForConditionalGeneration")
    missing_symbols = [symbol for symbol in required_symbols if not hasattr(transformers, symbol)]
    if missing_symbols:
        missing_display = ", ".join(missing_symbols)
        raise RuntimeError(
            "HunyuanOCR support requires a Transformers build with native Hunyuan support. "
            f"Missing symbols: {missing_display}. Current transformers version: {transformers.__version__}. "
            "Install hint: pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
        )


def ensure_deepseek_transformers_compat() -> None:
    if importlib.util.find_spec("addict") is None:
        raise RuntimeError(
            "DeepSeek-OCR-2 support requires the `addict` package for the model's remote code. "
            "Install hint: pip install addict"
        )


def ensure_nougat_dependencies() -> None:
    required_packages = ("nltk", "Levenshtein")
    missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
    if missing_packages:
        missing_display = ", ".join(missing_packages)
        raise RuntimeError(
            "Nougat post-processing requires additional packages. "
            f"Missing packages: {missing_display}. "
            "Install hint: pip install nltk python-Levenshtein"
        )


def resolve_deepseek_layout(model) -> str:
    if getattr(model.model, "qwen2_model", None) is None:
        raise RuntimeError("DeepSeek-OCR-2 did not expose the expected qwen2-backed vision stack.")
    return "ocr2"


def find_closest_deepseek_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def resolve_deepseek_crop_grid(
    height: int,
    width: int,
    *,
    crop_mode: bool,
    image_size: int,
    min_num: int = 2,
    max_num: int = 6,
) -> tuple[int, int]:
    if not crop_mode or (width <= image_size and height <= image_size):
        return (1, 1)

    target_ratios = sorted(
        {
            (grid_w, grid_h)
            for n in range(min_num, max_num + 1)
            for grid_w in range(1, n + 1)
            for grid_h in range(1, n + 1)
            if min_num <= grid_w * grid_h <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    return find_closest_deepseek_aspect_ratio(
        width / height,
        target_ratios,
        width,
        height,
        image_size,
    )


def square_pad_and_normalize_deepseek(
    image_tensor: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    height, width = image_tensor.shape[-2:]
    scale = min(image_size / height, image_size / width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    pad_h = image_size - resized_h
    pad_w = image_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=DEEPSEEK_NORMALIZE_MEAN)
    return (x - DEEPSEEK_NORMALIZE_MEAN) / DEEPSEEK_NORMALIZE_STD


def resize_and_split_deepseek_crops(
    image_tensor: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    image_size: int,
) -> torch.Tensor:
    grid_w, grid_h = grid_size
    target_width = image_size * grid_w
    target_height = image_size * grid_h
    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    resized = (resized - DEEPSEEK_NORMALIZE_MEAN) / DEEPSEEK_NORMALIZE_STD
    patches = resized.reshape(3, grid_h, image_size, grid_w, image_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(grid_h * grid_w, 3, image_size, image_size)
    return patches


def count_deepseek_image_tokens(state: dict) -> int:
    base_queries = math.ceil((state["base_size"] // DEEPSEEK_PATCH_SIZE) / DEEPSEEK_DOWNSAMPLE_RATIO)
    local_queries = math.ceil((state["image_size"] // DEEPSEEK_PATCH_SIZE) / DEEPSEEK_DOWNSAMPLE_RATIO)
    grid_w, grid_h = state["deepseek_crop_grid"]

    if state["crop_mode"]:
        image_token_count = base_queries * base_queries + 1
        if grid_w > 1 or grid_h > 1:
            image_token_count += (local_queries * grid_w) * (local_queries * grid_h)
        return image_token_count

    return local_queries * local_queries + 1


def build_deepseek_prompt_inputs(
    tokenizer,
    state: dict,
) -> tuple[str, dict[str, torch.Tensor], torch.Tensor]:
    prompt_text = state["prompt"]
    text_splits = prompt_text.split("<image>")
    if len(text_splits) != 2:
        raise RuntimeError("DeepSeek-OCR-2 prompts must contain exactly one <image> placeholder.")

    tokenized_ids: list[int] = []
    image_mask: list[bool] = []
    for text_part, is_image in ((text_splits[0], False), ("", True), (text_splits[1], False)):
        if is_image:
            image_token_count = count_deepseek_image_tokens(state)
            tokenized_ids.extend([DEEPSEEK_IMAGE_TOKEN_ID] * image_token_count)
            image_mask.extend([True] * image_token_count)
        else:
            text_token_ids = tokenizer.encode(text_part, add_special_tokens=False)
            tokenized_ids.extend(text_token_ids)
            image_mask.extend([False] * len(text_token_ids))

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    input_ids = torch.tensor([[bos_token_id, *tokenized_ids]], device=state["device"], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    images_seq_mask = torch.tensor([[False, *image_mask]], device=state["device"], dtype=torch.bool)
    return (
        prompt_text,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        images_seq_mask,
    )


def build_deepseek_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, Any]:
    global_image_size = state["base_size"] if state["crop_mode"] else state["image_size"]
    images_ori = square_pad_and_normalize_deepseek(image_tensor, global_image_size)
    images_ori = images_ori.to(device=state["device"], dtype=state["dtype"]).unsqueeze(0)

    grid_w, grid_h = state["deepseek_crop_grid"]
    if state["crop_mode"] and (grid_w > 1 or grid_h > 1):
        images_crop = resize_and_split_deepseek_crops(
            image_tensor,
            grid_size=state["deepseek_crop_grid"],
            image_size=state["image_size"],
        ).to(device=state["device"], dtype=state["dtype"])
    else:
        images_crop = torch.zeros(
            (1, 3, state["base_size"], state["base_size"]),
            device=state["device"],
            dtype=state["dtype"],
        )

    return {
        "images": [(images_crop, images_ori)],
        "images_seq_mask": state["deepseek_prompt_image_mask"].clone(),
        "images_spatial_crop": torch.tensor(
            [list(state["deepseek_crop_grid"])],
            device=state["device"],
            dtype=torch.long,
        ),
    }


def validate_deepseek_image_token_alignment(
    state: dict,
    vision_inputs: dict[str, Any],
) -> None:
    grid_w, grid_h = state["deepseek_crop_grid"]
    local_queries = math.ceil((state["image_size"] // DEEPSEEK_PATCH_SIZE) / DEEPSEEK_DOWNSAMPLE_RATIO)
    global_queries = math.ceil((state["base_size"] // DEEPSEEK_PATCH_SIZE) / DEEPSEEK_DOWNSAMPLE_RATIO)

    actual_feature_count = global_queries * global_queries + 1
    if state["crop_mode"] and (grid_w > 1 or grid_h > 1):
        patch_count = int(vision_inputs["images"][0][0].shape[0])
        actual_feature_count += patch_count * (local_queries * local_queries)

    expected_feature_count = int(state["deepseek_prompt_image_mask"].sum().item())
    if actual_feature_count != expected_feature_count:
        raise RuntimeError(
            "DeepSeek prompt/image token span mismatch: "
            f"prompt expects {expected_feature_count}, but preprocessing produced {actual_feature_count} visual tokens."
        )


def generate_deepseek_text(
    model,
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    prompt_token_count: int,
    vision_inputs: dict[str, Any],
    *,
    max_new_tokens: int,
) -> str:
    generated = model.generate(
        input_ids=prompt_model_inputs["input_ids"],
        attention_mask=prompt_model_inputs["attention_mask"],
        images=vision_inputs["images"],
        images_seq_mask=vision_inputs["images_seq_mask"],
        images_spatial_crop=vision_inputs["images_spatial_crop"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    new_tokens = generated[:, prompt_token_count:]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
        text = text[: -len(tokenizer.eos_token)]
    return text.strip()


def patch_deepseek_forward(model) -> None:
    if getattr(model, "_attack_forward_patched", False):
        return

    def forward_with_image_grad(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        sam_model = getattr(self, "sam_model", None)
        qwen2_model = getattr(self, "qwen2_model", None)

        if (
            sam_model is not None
            and qwen2_model is not None
            and images is not None
            and images_seq_mask is not None
            and (sequence_length != 1 or self.training)
            and torch.sum(images[0][1]).item() != 0
        ):
            updated_rows = []
            for row_embeds, image, image_mask in zip(inputs_embeds, images, images_seq_mask):
                patches, image_ori = image
                if torch.sum(patches).item() != 0:
                    local_features = self.projector(qwen2_model(sam_model(patches)))
                    local_features = local_features.reshape(-1, local_features.shape[-1])
                    global_features = self.projector(qwen2_model(sam_model(image_ori)))
                    global_features = global_features.reshape(-1, global_features.shape[-1])
                    view_separator = self.view_seperator[None, :].to(local_features.dtype)
                    image_features = torch.cat([local_features, global_features, view_separator], dim=0)
                else:
                    global_features = self.projector(qwen2_model(sam_model(image_ori)))
                    global_features = global_features.reshape(-1, global_features.shape[-1])
                    view_separator = self.view_seperator[None, :].to(global_features.dtype)
                    image_features = torch.cat([global_features, view_separator], dim=0)

                updated_rows.append(
                    row_embeds.masked_scatter(
                        image_mask.unsqueeze(-1).to(device=row_embeds.device),
                        image_features.to(device=row_embeds.device, dtype=row_embeds.dtype),
                    )
                )
            inputs_embeds = torch.stack(updated_rows, dim=0)

        parent_forward = self.__class__.__mro__[1].forward
        return parent_forward(
            self,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    model.model.forward = MethodType(forward_with_image_grad, model.model)
    model._attack_forward_patched = True


def build_qianfan_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    image_inputs = state["image_processor"](
        images=[image_tensor],
        return_tensors="pt",
        do_rescale=False,
        crop_to_patches=True,
        min_patches=state["min_patches"],
        max_patches=state["max_patches"],
    )
    return {
        "pixel_values": image_inputs["pixel_values"].to(device=state["device"], dtype=state["dtype"]),
    }


def resolve_hunyuan_resize_mode(image_processor) -> str:
    resample = getattr(image_processor, "resample", None)
    resample_name = getattr(resample, "name", None)
    if isinstance(resample_name, str):
        normalized_name = resample_name.lower()
    elif isinstance(resample, str):
        normalized_name = resample.lower()
    elif isinstance(resample, int):
        normalized_name = {
            0: "nearest",
            2: "bilinear",
            3: "bicubic",
        }.get(resample, "bicubic")
    else:
        normalized_name = "bicubic"

    if normalized_name in {"nearest", "nearest_exact"}:
        return "nearest"
    if normalized_name in {"bilinear", "linear"}:
        return "bilinear"
    return "bicubic"


def smart_resize_hunyuan(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise RuntimeError(f"Hunyuan absolute aspect ratio must be smaller than 200, got {aspect_ratio}.")

    resized_h = max(factor, int(round(height / factor) * factor))
    resized_w = max(factor, int(round(width / factor) * factor))

    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = max(factor, math.ceil(height * beta / factor) * factor)
        resized_w = max(factor, math.ceil(width * beta / factor) * factor)

    return resized_h, resized_w


def build_hunyuan_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    patch_size = state["patch_size"]
    merge_size = state["merge_size"]
    temporal_patch_size = state["temporal_patch_size"]
    if temporal_patch_size != 1:
        raise RuntimeError(
            "This OCR script expects Hunyuan image preprocessing with temporal_patch_size=1, "
            f"got {temporal_patch_size}."
        )

    resized_h, resized_w = smart_resize_hunyuan(
        int(image_tensor.shape[-2]),
        int(image_tensor.shape[-1]),
        factor=patch_size * merge_size,
        min_pixels=state["min_pixels"],
        max_pixels=state["max_pixels"],
    )

    resize_kwargs = {}
    if state["resize_mode"] in {"bilinear", "bicubic"}:
        resize_kwargs["align_corners"] = False
    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode=state["resize_mode"],
        **resize_kwargs,
    ).squeeze(0)
    x = (x - state["mean"].view(3, 1, 1)) / state["std"].view(3, 1, 1)

    grid_t = 1
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise RuntimeError(
            "Hunyuan resized image dimensions were not divisible by the merge size after patching: "
            f"grid_h={grid_h}, grid_w={grid_w}, merge_size={merge_size}."
        )

    channels = x.shape[0]
    patches = x.reshape(
        1,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 3, 5, 6, 1, 4, 7)
    pixel_values = patches.reshape(grid_t * grid_h * grid_w, channels * patch_size * patch_size)
    return {
        "pixel_values": pixel_values.to(device=state["device"], dtype=state["dtype"]),
        "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]], device=state["device"], dtype=torch.long),
    }


def build_encoder_decoder_prompt_inputs(
    tokenizer,
    prompt: str | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if prompt is None:
        return {}

    decoder_prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    return {"decoder_prompt_ids": decoder_prompt_ids}


def build_encoder_decoder_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    if is_encoder_decoder_family(state["model_family"]):
        image_processor = state["image_processor"]
        size = image_processor.size
        target_h = int(size["height"])
        target_w = int(size["width"])

        x = image_tensor
        if state["model_family"] == "encoder_decoder_nougat" and getattr(image_processor, "do_crop_margin", False):
            gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
            gray_min = gray.amin()
            gray_max = gray.amax()
            if float(gray_max.item()) != float(gray_min.item()):
                gray = (gray - gray_min) / (gray_max - gray_min) * 255.0
                coords = (gray < 200).nonzero(as_tuple=False)
                if coords.numel() > 0:
                    y_min = int(coords[:, 0].min().item())
                    y_max = int(coords[:, 0].max().item()) + 1
                    x_min = int(coords[:, 1].min().item())
                    x_max = int(coords[:, 1].max().item()) + 1
                    x = x[:, y_min:y_max, x_min:x_max]

        if getattr(image_processor, "do_align_long_axis", False):
            input_h, input_w = int(x.shape[-2]), int(x.shape[-1])
            if (target_w < target_h and input_w > input_h) or (target_w > target_h and input_w < input_h):
                x = torch.rot90(x, k=3, dims=(-2, -1))

        if getattr(image_processor, "do_resize", True):
            input_h, input_w = int(x.shape[-2]), int(x.shape[-1])
            shortest_edge = min(target_h, target_w)
            if input_w <= input_h:
                resized_h = int(shortest_edge * input_h / input_w)
                resized_w = shortest_edge
            else:
                resized_h = shortest_edge
                resized_w = int(shortest_edge * input_w / input_h)
            x = F.interpolate(
                x.unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

        if getattr(image_processor, "do_thumbnail", True):
            input_h, input_w = int(x.shape[-2]), int(x.shape[-1])
            resized_h = min(input_h, target_h)
            resized_w = min(input_w, target_w)
            if resized_h != input_h or resized_w != input_w:
                if input_h > input_w:
                    resized_w = int(input_w * resized_h / input_h)
                elif input_w > input_h:
                    resized_h = int(input_h * resized_w / input_w)
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)

        if getattr(image_processor, "do_pad", True):
            input_h, input_w = int(x.shape[-2]), int(x.shape[-1])
            pad_h = target_h - input_h
            pad_w = target_w - input_w
            if pad_h < 0 or pad_w < 0:
                raise RuntimeError(
                    "Encoder-decoder preprocessing produced an image larger than the target canvas: "
                    f"{input_h}x{input_w} > {target_h}x{target_w}."
                )
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        if getattr(image_processor, "do_normalize", True):
            mean = torch.tensor(image_processor.image_mean, device=x.device, dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor(image_processor.image_std, device=x.device, dtype=x.dtype).view(3, 1, 1)
            x = (x - mean) / std

        return {
            "pixel_values": x.unsqueeze(0).to(device=state["device"], dtype=state["dtype"]),
        }

    image_inputs = state["image_processor"](
        images=[image_tensor],
        return_tensors="pt",
        do_rescale=False,
    )
    return {
        "pixel_values": image_inputs["pixel_values"].to(device=state["device"], dtype=state["dtype"]),
    }


def generate_encoder_decoder_text(
    model,
    processor,
    state: dict,
    vision_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> str:
    tokenizer = processor.tokenizer
    bad_words_ids = [[tokenizer.unk_token_id]] if tokenizer.unk_token_id is not None else None
    generation_kwargs = {
        "pixel_values": vision_inputs["pixel_values"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "bad_words_ids": bad_words_ids,
        "return_dict_in_generate": True,
        "do_sample": False,
    }

    decoder_prompt_ids = state.get("decoder_prompt_ids")
    if decoder_prompt_ids is not None:
        outputs = model.generate(
            decoder_input_ids=decoder_prompt_ids,
            **generation_kwargs,
        )
        generated_ids = outputs.sequences[:, decoder_prompt_ids.shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    outputs = model.generate(**generation_kwargs)
    generation = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    if state["model_family"] == "encoder_decoder_nougat":
        return processor.post_process_generation(generation, fix_markdown=True).strip()
    return generation.strip()


def build_encoder_decoder_teacher_forced_batch(
    tokenizer,
    prompt_model_inputs: dict[str, torch.Tensor],
    text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (target_ids.shape[1] == 0 or target_ids[0, -1].item() != eos_token_id):
        eos_tensor = torch.tensor([[eos_token_id]], device=device, dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    decoder_prompt_ids = prompt_model_inputs.get("decoder_prompt_ids")
    if decoder_prompt_ids is not None:
        decoder_input_ids = torch.cat([decoder_prompt_ids, target_ids[:, :-1]], dim=1)
        ignored_prompt_labels = torch.full(
            (decoder_prompt_ids.shape[0], decoder_prompt_ids.shape[1] - 1),
            -100,
            device=device,
            dtype=target_ids.dtype,
        )
        labels = torch.cat([ignored_prompt_labels, target_ids], dim=1)
    else:
        decoder_start_token_id = tokenizer.bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
        if decoder_start_token_id is None:
            raise RuntimeError("Encoder-decoder OCR tokenizer does not expose a decoder start token.")

        decoder_start = torch.tensor([[decoder_start_token_id]], device=device, dtype=target_ids.dtype)
        decoder_input_ids = torch.cat([decoder_start, target_ids[:, :-1]], dim=1)
        labels = target_ids

    return {
        "model_inputs": {
            "decoder_input_ids": decoder_input_ids,
        },
        "labels": labels,
    }


def build_model_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, Any]:
    if state["model_family"] == "deepseek":
        return build_deepseek_vision_inputs(state, image_tensor)
    if state["model_family"] == "qianfan":
        return build_qianfan_vision_inputs(state, image_tensor)
    if state["model_family"] == "hunyuan":
        return build_hunyuan_vision_inputs(state, image_tensor)
    if is_encoder_decoder_family(state["model_family"]):
        return build_encoder_decoder_vision_inputs(state, image_tensor)
    return build_qwen_vision_inputs(state, image_tensor)


def validate_hunyuan_preprocessing(
    processor,
    prompt_text: str,
    prompt_model_inputs: dict[str, torch.Tensor],
    state: dict,
    source_image_path: Path,
    x_clean: torch.Tensor,
) -> None:
    with Image.open(source_image_path) as image:
        reference_inputs = processor(
            text=[prompt_text],
            images=[image.convert("RGB")],
            return_tensors="pt",
        )

    reference_input_ids = reference_inputs["input_ids"]
    prompt_input_ids = prompt_model_inputs["input_ids"].detach().cpu()
    if not torch.equal(reference_input_ids, prompt_input_ids):
        raise RuntimeError(
            "Hunyuan prompt tokenization mismatch between the dummy prompt image and the source image. "
            "Ensure prompt_dummy_image_size matches the source image dimensions."
        )

    reference_attention_mask = reference_inputs.get("attention_mask")
    prompt_attention_mask = prompt_model_inputs.get("attention_mask")
    if (
        reference_attention_mask is not None
        and prompt_attention_mask is not None
        and not torch.equal(reference_attention_mask, prompt_attention_mask.detach().cpu())
    ):
        raise RuntimeError("Hunyuan attention mask mismatch between dummy prompt preprocessing and the source image.")

    manual_vision_inputs = build_hunyuan_vision_inputs(state, x_clean.squeeze(0))
    reference_grid = reference_inputs["image_grid_thw"]
    manual_grid = manual_vision_inputs["image_grid_thw"].detach().cpu()
    if not torch.equal(reference_grid, manual_grid):
        raise RuntimeError(
            "Hunyuan image_grid_thw mismatch between the processor and manual preprocessing: "
            f"processor={reference_grid.tolist()} manual={manual_grid.tolist()}."
        )

    reference_pixels = reference_inputs["pixel_values"]
    manual_pixels = manual_vision_inputs["pixel_values"].detach().cpu().to(dtype=reference_pixels.dtype)
    if reference_pixels.shape != manual_pixels.shape:
        raise RuntimeError(
            "Hunyuan pixel_values shape mismatch between the processor and manual preprocessing: "
            f"processor={tuple(reference_pixels.shape)} manual={tuple(manual_pixels.shape)}."
        )

    diff = (manual_pixels - reference_pixels).abs()
    if not torch.isfinite(diff).all():
        raise RuntimeError("Hunyuan preprocessing sanity check produced non-finite pixel differences.")

    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
    if max_abs_diff > 0.25 and mean_abs_diff > 0.03:
        raise RuntimeError(
            "Hunyuan manual preprocessing diverged from the processor output: "
            f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}."
        )


def count_qianfan_prompt_image_patches(
    prompt_model_inputs: dict[str, torch.Tensor],
    processor,
) -> int:
    tokenizer = processor.tokenizer
    start_image_token_id = getattr(tokenizer, "start_image_token_id", None)
    end_image_token_id = getattr(tokenizer, "end_image_token_id", None)
    context_image_token_id = getattr(tokenizer, "context_image_token_id", None)
    image_seq_length = getattr(processor, "image_seq_length", None)

    if None in (start_image_token_id, end_image_token_id, context_image_token_id, image_seq_length):
        raise RuntimeError("Qianfan-OCR processor/tokenizer does not expose the expected image token metadata.")

    input_ids = prompt_model_inputs["input_ids"][0]
    start_positions = (input_ids == start_image_token_id).nonzero(as_tuple=False).flatten()
    end_positions = (input_ids == end_image_token_id).nonzero(as_tuple=False).flatten()
    if start_positions.numel() != 1 or end_positions.numel() != 1:
        raise RuntimeError("Expected exactly one Qianfan image placeholder span in the OCR prompt.")

    start_index = int(start_positions[0].item())
    end_index = int(end_positions[0].item())
    if end_index <= start_index:
        raise RuntimeError("Invalid Qianfan image placeholder span in the OCR prompt.")

    context_token_count = int((input_ids[start_index + 1 : end_index] == context_image_token_id).sum().item())
    if context_token_count % image_seq_length != 0:
        raise RuntimeError(
            "Qianfan prompt image token count was not divisible by the expected image sequence length."
        )

    return context_token_count // image_seq_length


def transcription_loss(
    model,
    teacher_forced_batch: dict[str, torch.Tensor],
    vision_inputs: dict[str, Any],
) -> torch.Tensor:
    images_seq_mask = vision_inputs.get("images_seq_mask")
    if images_seq_mask is not None:
        full_sequence_length = teacher_forced_batch["model_inputs"]["input_ids"].shape[1]
        current_mask_length = images_seq_mask.shape[1]
        if current_mask_length > full_sequence_length:
            raise RuntimeError(
                "DeepSeek images_seq_mask was longer than the teacher-forced sequence: "
                f"{current_mask_length} > {full_sequence_length}."
            )
        if current_mask_length < full_sequence_length:
            vision_inputs = dict(vision_inputs)
            pad_width = full_sequence_length - current_mask_length
            vision_inputs["images_seq_mask"] = torch.cat(
                [
                    images_seq_mask,
                    torch.zeros(
                        (images_seq_mask.shape[0], pad_width),
                        device=images_seq_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )

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
    *,
    epsilon: float = EPSILON,
    alpha: float = ALPHA,
    steps: int = STEPS,
    random_start: bool = RANDOM_START,
    event_callback=None,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    delta = torch.zeros_like(x_clean, dtype=torch.float32)
    if random_start:
        delta.uniform_(-epsilon, epsilon)
        project_delta(delta, x_clean, epsilon)
    delta.requires_grad_(True)

    # Guard against silent failures where quantization or preprocessing blocks gradient flow.
    saw_nonzero_grad = False
    last_loss = 0.0
    last_grad_inf = 0.0
    progress = tqdm(range(steps))
    for step_index in progress:
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        vision_inputs = build_model_vision_inputs(state, x_adv.squeeze(0))
        loss = transcription_loss(model, teacher_forced_batch, vision_inputs)
        loss.backward()

        if delta.grad is None:
            raise RuntimeError("Expected PGD gradients on the perturbation tensor.")

        grad = delta.grad.detach()
        grad_inf = float(grad.abs().max().item())
        saw_nonzero_grad = saw_nonzero_grad or grad_inf > 0.0

        with torch.no_grad():
            # PGD ascent on transcription loss, then project back into the L_inf epsilon ball.
            delta.add_(alpha * grad.sign())
            project_delta(delta, x_clean, epsilon)

        last_loss = float(loss.item())
        last_grad_inf = grad_inf
        progress.set_postfix(loss=f"{last_loss:.4f}", grad_inf=f"{last_grad_inf:.6f}")
        if event_callback is not None and (step_index == 0 or step_index == steps - 1 or (step_index + 1) % 10 == 0):
            event_callback(
                "optimization_metric",
                step=step_index + 1,
                loss=last_loss,
                gradient_inf=last_grad_inf,
            )

    if not saw_nonzero_grad:
        # Failing early here avoids writing artifacts from a no-op attack.
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
    run_config: dict,
    *,
    final_loss: float,
    final_grad_inf: float,
    linf_delta: float,
) -> str:
    edit_distance = levenshtein_distance(clean_text, adv_text)
    normalized_edit_rate = edit_distance / max(1, len(clean_text))
    exact_match = clean_text == adv_text

    lines = [
        f"Model: {run_config['model_name']}",
        f"Model key: {run_config['model_key']}",
        f"Prompt: {run_config['ocr_prompt']}",
        f"Source image: {run_config['source_image_path'].resolve()}",
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
        f"Adversarial image path: {run_config['output_adv_path'].resolve()}",
    ]
    return "\n".join(lines)


def _legacy_main() -> None:
    run_config = build_run_config()
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Repo root: {REPO_ROOT}")
    print(f"Source image: {run_config['source_image_path']}")
    print(f"Model: {run_config['model_name']}")
    print(f"Model key: {run_config['model_key']}")
    print(f"Model family: {run_config['model_family']}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")

    run_config["output_adv_path"].parent.mkdir(parents=True, exist_ok=True)
    run_config["output_report_path"].parent.mkdir(parents=True, exist_ok=True)

    x_clean = load_image_tensor(run_config["source_image_path"], device)
    tokenizer = None
    processor = None
    if run_config["model_family"] == "deepseek":
        ensure_deepseek_transformers_compat()
        tokenizer = AutoTokenizer.from_pretrained(run_config["model_name"], trust_remote_code=True)
        model = AutoModel.from_pretrained(
            run_config["model_name"],
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=model_dtype,
        ).to(device)
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": run_config["model_family"],
            "prompt": run_config["ocr_prompt"],
            "base_size": run_config["base_size"],
            "image_size": run_config["image_size"],
            "crop_mode": run_config["crop_mode"],
            "deepseek_layout": resolve_deepseek_layout(model),
            "deepseek_crop_grid": resolve_deepseek_crop_grid(
                int(x_clean.shape[-2]),
                int(x_clean.shape[-1]),
                crop_mode=run_config["crop_mode"],
                image_size=run_config["image_size"],
            ),
        }
        if state["deepseek_layout"] != "ocr2":
            raise RuntimeError(f"Unsupported DeepSeek layout: {state['deepseek_layout']}")
        prompt_text, prompt_model_inputs, prompt_image_mask = build_deepseek_prompt_inputs(tokenizer, state)
        state["deepseek_prompt_image_mask"] = prompt_image_mask
        prompt_token_count = prompt_model_inputs["input_ids"].shape[1]
        patch_deepseek_forward(model)
    elif run_config["model_family"] == "qianfan":
        ensure_qianfan_transformers_support()
        processor = AutoProcessor.from_pretrained(run_config["model_name"])
        model = AutoModelForImageTextToText.from_pretrained(
            run_config["model_name"],
            torch_dtype=model_dtype,
        ).to(device)
        image_processor = processor.image_processor
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": run_config["model_family"],
            "image_processor": image_processor,
            "min_patches": 1,
            "max_patches": 12,
        }
        prompt_dummy_image_size = (
            int(x_clean.shape[-1]),
            int(x_clean.shape[-2]),
        )
    elif run_config["model_family"] == "hunyuan":
        ensure_hunyuan_transformers_support()
        processor = AutoProcessor.from_pretrained(
            run_config["model_name"],
            use_fast=False,
        )
        hunyuan_model_cls = getattr(transformers, "HunYuanVLForConditionalGeneration")
        model = hunyuan_model_cls.from_pretrained(
            run_config["model_name"],
            attn_implementation="eager",
            dtype=model_dtype,
        ).to(device)
        image_processor = processor.image_processor
        vision_config = model.config.vision_config
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": run_config["model_family"],
            "min_pixels": int(getattr(image_processor, "min_pixels")),
            "max_pixels": int(getattr(image_processor, "max_pixels")),
            "patch_size": int(getattr(image_processor, "patch_size", vision_config.patch_size)),
            "temporal_patch_size": int(
                getattr(image_processor, "temporal_patch_size", getattr(vision_config, "temporal_patch_size", 1))
            ),
            "merge_size": int(getattr(image_processor, "merge_size", vision_config.spatial_merge_size)),
            "resize_mode": resolve_hunyuan_resize_mode(image_processor),
            "mean": torch.tensor(image_processor.image_mean, device=device, dtype=torch.float32),
            "std": torch.tensor(image_processor.image_std, device=device, dtype=torch.float32),
        }
        prompt_dummy_image_size = (
            int(x_clean.shape[-1]),
            int(x_clean.shape[-2]),
        )
    elif is_encoder_decoder_family(run_config["model_family"]):
        if run_config["model_family"] == "encoder_decoder_nougat":
            ensure_nougat_dependencies()
            processor = NougatProcessor.from_pretrained(run_config["model_name"], backend="torchvision")
        else:
            processor = DonutProcessor.from_pretrained(
                run_config["model_name"],
                backend="torchvision",
                use_fast=False,
            )

        model = VisionEncoderDecoderModel.from_pretrained(
            run_config["model_name"],
            torch_dtype=model_dtype,
        ).to(device)
        prompt_model_inputs = build_encoder_decoder_prompt_inputs(
            processor.tokenizer,
            run_config["ocr_prompt"],
            device,
        )
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": run_config["model_family"],
            "image_processor": processor.image_processor,
            **prompt_model_inputs,
        }
    else:
        processor = AutoProcessor.from_pretrained(run_config["model_name"], trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            run_config["model_name"],
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(device)
        vision_config = model.config.vision_config
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": run_config["model_family"],
            "model_input_size": MODEL_INPUT_SIZE,
            "patch_size": vision_config.patch_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            "merge_size": vision_config.spatial_merge_size,
            "mean": torch.tensor(processor.image_processor.image_mean, device=device, dtype=torch.float32),
            "std": torch.tensor(processor.image_processor.image_std, device=device, dtype=torch.float32),
        }
        prompt_dummy_image_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    model.eval()
    model.requires_grad_(False)

    if run_config["model_family"] != "deepseek" and not is_encoder_decoder_family(run_config["model_family"]):
        prompt_text, prompt_model_inputs = build_chat_prompt_inputs(
            processor,
            device,
            run_config["ocr_prompt"],
            prompt_dummy_image_size,
        )
        if run_config["model_family"] == "hunyuan":
            validate_hunyuan_preprocessing(
                processor,
                prompt_text,
                prompt_model_inputs,
                state,
                run_config["source_image_path"],
                x_clean,
            )
        prompt_token_count = prompt_model_inputs["input_ids"].shape[1]

    clean_vision_inputs = build_model_vision_inputs(state, x_clean.squeeze(0))
    if run_config["model_family"] == "deepseek":
        validate_deepseek_image_token_alignment(state, clean_vision_inputs)
    if run_config["model_family"] == "qianfan":
        expected_patch_count = count_qianfan_prompt_image_patches(prompt_model_inputs, processor)
        actual_patch_count = int(clean_vision_inputs["pixel_values"].shape[0])
        if expected_patch_count != actual_patch_count:
            raise RuntimeError(
                "Qianfan prompt/image patch count mismatch: "
                f"prompt expects {expected_patch_count}, but preprocessing produced {actual_patch_count}."
            )
    with torch.no_grad():
        if run_config["model_family"] == "deepseek":
            clean_text = generate_deepseek_text(
                model,
                tokenizer,
                prompt_model_inputs,
                prompt_token_count,
                clean_vision_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        elif is_encoder_decoder_family(run_config["model_family"]):
            clean_text = generate_encoder_decoder_text(
                model,
                processor,
                state,
                clean_vision_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        else:
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

    if is_encoder_decoder_family(run_config["model_family"]):
        teacher_forced_batch = build_encoder_decoder_teacher_forced_batch(
            processor.tokenizer,
            prompt_model_inputs,
            # Reuse the clean transcript so PGD can explicitly maximize error against the model's own baseline text.
            clean_text,
            device,
        )
    else:
        teacher_forced_batch = build_teacher_forced_batch(
            tokenizer if tokenizer is not None else processor.tokenizer,
            prompt_model_inputs,
            # Reuse the clean transcript so PGD can explicitly maximize error against the model's own baseline text.
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

    adv_vision_inputs = build_model_vision_inputs(state, x_adv.squeeze(0))
    with torch.no_grad():
        if run_config["model_family"] == "deepseek":
            adv_text = generate_deepseek_text(
                model,
                tokenizer,
                prompt_model_inputs,
                prompt_token_count,
                adv_vision_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        elif is_encoder_decoder_family(run_config["model_family"]):
            adv_text = generate_encoder_decoder_text(
                model,
                processor,
                state,
                adv_vision_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        else:
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

    save_image_tensor(x_adv, run_config["output_adv_path"])
    run_config["output_report_path"].write_text(
        build_report(
            clean_text,
            adv_text,
            run_config,
            final_loss=final_loss,
            final_grad_inf=final_grad_inf,
            linf_delta=linf_delta,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Clean OCR: {clean_text}")
    print(f"Adversarial OCR: {adv_text}")
    print(f"Perturbation L_inf: {linf_delta:.6f}")
    print(f"Saved adversarial image to {run_config['output_adv_path'].resolve()}")
    print(f"Saved report to {run_config['output_report_path'].resolve()}")


def _ocr_model_keys(config: dict) -> tuple[str, list[str]]:
    selected = config.get("models", config.get("model_key"))
    if isinstance(selected, str):
        return selected, [selected]
    if isinstance(selected, list):
        if not selected:
            raise ValueError("At least one OCR model must be selected.")
        return selected[0], list(selected)
    if isinstance(selected, dict):
        attack_key = selected.get("attack")
        transfer = selected.get("transfer", [])
        if not isinstance(attack_key, str):
            raise ValueError("OCR pipeline models.attack must be a model key.")
        if isinstance(transfer, str):
            transfer = [transfer]
        if not isinstance(transfer, list):
            raise ValueError("OCR pipeline models.transfer must be a list.")
        return attack_key, [attack_key, *transfer]
    raise ValueError("OCR models must be a model key, list, or pipeline object.")


def _resolve_ocr_prompt(
    config: dict,
    model_key: str,
    *,
    inference: bool = False,
    allow_global_prompt: bool = False,
) -> str | None:
    model_definition = MODEL_CONFIGS[model_key]
    if inference:
        inference_prompts = config.get("inference_prompts")
        if isinstance(inference_prompts, dict):
            if model_key in inference_prompts:
                return inference_prompts[model_key]
            model_name = model_definition["model_name"]
            if model_name in inference_prompts:
                return inference_prompts[model_name]
        if allow_global_prompt and config.get("prompt") is not None:
            return config.get("prompt")
        return model_definition.get("ocr_prompt")
    return config.get("prompt") if config.get("prompt") is not None else model_definition.get("ocr_prompt")


def _ocr_config_for_model(
    config: dict,
    model_key: str,
    *,
    inference: bool = False,
    allow_global_prompt: bool = False,
) -> dict:
    model_definition = MODEL_CONFIGS[model_key]
    model_config = dict(config)
    model_config["model_key"] = model_key
    model_config["model_name"] = model_definition["model_name"]
    model_config["model_family"] = model_definition["model_family"]
    model_config["ocr_prompt"] = _resolve_ocr_prompt(
        config,
        model_key,
        inference=inference,
        allow_global_prompt=allow_global_prompt,
    )
    for key in ("base_size", "image_size", "crop_mode"):
        if key in model_definition:
            model_config[key] = model_definition[key]
    model_config["model_input_size"] = int(config.get("model_input_size", MODEL_INPUT_SIZE))
    return model_config


def _pretrained_kwargs(config: dict, **kwargs: Any) -> dict[str, Any]:
    """Add a configured model revision to every loader call, when supplied."""
    result = dict(kwargs)
    revisions = config.get("revisions")
    revision = None
    if isinstance(revisions, dict):
        revision = revisions.get(config.get("model_key"))
        if revision is None:
            revision = revisions.get(config.get("model_name"))
    if revision is None:
        revision = config.get("revision")
    if revision is not None:
        result["revision"] = revision
    return result


def _release_ocr_memory() -> None:
    """Release Python references before asking CUDA to reclaim cached blocks."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _load_ocr_runtime(config: dict, x_clean: torch.Tensor, event_callback=None) -> dict:
    device = torch.device(config.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    input_paths = config["inputs"]
    source_image_value = input_paths.get("clean", input_paths.get("image"))
    if source_image_value is None:
        raise ValueError("OCR runtime requires inputs.image or inputs.clean.")
    source_image_path = Path(source_image_value)
    model_family = config["model_family"]
    model_name = config["model_name"]
    prompt = config.get("ocr_prompt")
    if event_callback is not None:
        event_callback("model_started", model_key=config["model_key"], model_name=model_name, device=str(device))

    tokenizer = None
    processor = None
    prompt_model_inputs = {}
    prompt_token_count = 0
    if model_family == "deepseek":
        ensure_deepseek_transformers_compat()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **_pretrained_kwargs(config, trust_remote_code=True)
        )
        model = AutoModel.from_pretrained(
            model_name,
            **_pretrained_kwargs(
                config,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=model_dtype,
            ),
        ).to(device)
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": model_family,
            "prompt": prompt,
            "base_size": int(config.get("base_size", 1024)),
            "image_size": int(config.get("image_size", 768)),
            "crop_mode": bool(config.get("crop_mode", True)),
            "deepseek_layout": resolve_deepseek_layout(model),
            "deepseek_crop_grid": resolve_deepseek_crop_grid(
                int(x_clean.shape[-2]),
                int(x_clean.shape[-1]),
                crop_mode=bool(config.get("crop_mode", True)),
                image_size=int(config.get("image_size", 768)),
            ),
        }
        if state["deepseek_layout"] != "ocr2":
            raise RuntimeError(f"Unsupported DeepSeek layout: {state['deepseek_layout']}")
        prompt_text, prompt_model_inputs, prompt_image_mask = build_deepseek_prompt_inputs(tokenizer, state)
        state["deepseek_prompt_image_mask"] = prompt_image_mask
        prompt_token_count = prompt_model_inputs["input_ids"].shape[1]
        patch_deepseek_forward(model)
    elif model_family == "qianfan":
        ensure_qianfan_transformers_support()
        processor = AutoProcessor.from_pretrained(model_name, **_pretrained_kwargs(config))
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, **_pretrained_kwargs(config, torch_dtype=model_dtype)
        ).to(device)
        image_processor = processor.image_processor
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": model_family,
            "image_processor": image_processor,
            "min_patches": 1,
            "max_patches": 12,
        }
        prompt_dummy_image_size = (int(x_clean.shape[-1]), int(x_clean.shape[-2]))
    elif model_family == "hunyuan":
        ensure_hunyuan_transformers_support()
        processor = AutoProcessor.from_pretrained(model_name, **_pretrained_kwargs(config, use_fast=False))
        model = getattr(transformers, "HunYuanVLForConditionalGeneration").from_pretrained(
            model_name, **_pretrained_kwargs(config, attn_implementation="eager", dtype=model_dtype)
        ).to(device)
        image_processor = processor.image_processor
        vision_config = model.config.vision_config
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": model_family,
            "min_pixels": int(getattr(image_processor, "min_pixels")),
            "max_pixels": int(getattr(image_processor, "max_pixels")),
            "patch_size": int(getattr(image_processor, "patch_size", vision_config.patch_size)),
            "temporal_patch_size": int(getattr(image_processor, "temporal_patch_size", getattr(vision_config, "temporal_patch_size", 1))),
            "merge_size": int(getattr(image_processor, "merge_size", vision_config.spatial_merge_size)),
            "resize_mode": resolve_hunyuan_resize_mode(image_processor),
            "mean": torch.tensor(image_processor.image_mean, device=device, dtype=torch.float32),
            "std": torch.tensor(image_processor.image_std, device=device, dtype=torch.float32),
        }
        prompt_dummy_image_size = (int(x_clean.shape[-1]), int(x_clean.shape[-2]))
    elif is_encoder_decoder_family(model_family):
        if model_family == "encoder_decoder_nougat":
            ensure_nougat_dependencies()
            processor = NougatProcessor.from_pretrained(
                model_name, **_pretrained_kwargs(config, backend="torchvision")
            )
        else:
            processor = DonutProcessor.from_pretrained(
                model_name, **_pretrained_kwargs(config, backend="torchvision", use_fast=False)
            )
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name, **_pretrained_kwargs(config, torch_dtype=model_dtype)
        ).to(device)
        prompt_model_inputs = build_encoder_decoder_prompt_inputs(processor.tokenizer, prompt, device)
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": model_family,
            "image_processor": processor.image_processor,
            **prompt_model_inputs,
        }
    else:
        processor = AutoProcessor.from_pretrained(
            model_name, **_pretrained_kwargs(config, trust_remote_code=True)
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, **_pretrained_kwargs(config, trust_remote_code=True, torch_dtype=model_dtype)
        ).to(device)
        vision_config = model.config.vision_config
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": model_family,
            "model_input_size": int(config.get("model_input_size", MODEL_INPUT_SIZE)),
            "patch_size": vision_config.patch_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            "merge_size": vision_config.spatial_merge_size,
            "mean": torch.tensor(processor.image_processor.image_mean, device=device, dtype=torch.float32),
            "std": torch.tensor(processor.image_processor.image_std, device=device, dtype=torch.float32),
        }
        prompt_dummy_image_size = (state["model_input_size"], state["model_input_size"])

    model.eval()
    model.requires_grad_(False)
    if model_family != "deepseek" and not is_encoder_decoder_family(model_family):
        prompt_text, prompt_model_inputs = build_chat_prompt_inputs(
            processor, device, prompt, prompt_dummy_image_size
        )
        if model_family == "hunyuan":
            validate_hunyuan_preprocessing(
                processor, prompt_text, prompt_model_inputs, state, source_image_path, x_clean
            )
        prompt_token_count = prompt_model_inputs["input_ids"].shape[1]

    clean_vision_inputs = build_model_vision_inputs(state, x_clean.squeeze(0))
    if model_family == "deepseek":
        validate_deepseek_image_token_alignment(state, clean_vision_inputs)
    if model_family == "qianfan":
        expected = count_qianfan_prompt_image_patches(prompt_model_inputs, processor)
        actual = int(clean_vision_inputs["pixel_values"].shape[0])
        if expected != actual:
            raise RuntimeError(f"Qianfan prompt/image patch count mismatch: prompt expects {expected}, manual preprocessing produced {actual}.")
    return {
        "config": config,
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "state": state,
        "prompt_model_inputs": prompt_model_inputs,
        "prompt_token_count": prompt_token_count,
        "device": device,
    }


def _generate_ocr_text(runtime: dict, image_tensor: torch.Tensor, max_new_tokens: int) -> str:
    state = runtime["state"]
    vision_inputs = build_model_vision_inputs(state, image_tensor.squeeze(0))
    with torch.no_grad():
        if state["model_family"] == "deepseek":
            return generate_deepseek_text(
                runtime["model"], runtime["tokenizer"], runtime["prompt_model_inputs"], runtime["prompt_token_count"], vision_inputs, max_new_tokens=max_new_tokens
            )
        if is_encoder_decoder_family(state["model_family"]):
            return generate_encoder_decoder_text(
                runtime["model"], runtime["processor"], state, vision_inputs, max_new_tokens=max_new_tokens
            )
        return generate_greedy_text(
            runtime["model"], runtime["processor"], runtime["prompt_model_inputs"], runtime["prompt_token_count"], vision_inputs, max_new_tokens=max_new_tokens
        )


def _run_one_ocr_inference(config: dict, model_key: str, image_paths: dict[str, Path], event_callback=None) -> dict:
    model_config = _ocr_config_for_model(
        config,
        model_key,
        inference=True,
        allow_global_prompt=bool(config.get("_allow_global_inference_prompt", False)),
    )
    device = torch.device(model_config.get("device", "cuda:0"))
    tensors = {}
    runtime = None
    try:
        tensors = {key: load_image_tensor(path, device) for key, path in image_paths.items()}
        base_key = "clean" if "clean" in tensors else "image"
        runtime = _load_ocr_runtime(model_config, tensors[base_key], event_callback)
        outputs = {
            key: _generate_ocr_text(runtime, tensor, int(config["generation"]["max_new_tokens"]))
            for key, tensor in tensors.items()
        }
        if event_callback is not None:
            event_callback(
                "model_completed",
                model_key=model_key,
                output_lengths={key: len(value) for key, value in outputs.items()},
            )
        return outputs
    finally:
        # Explicitly drop model, processor, and GPU tensor references before the
        # next transfer model is loaded.  empty_cache is harmless on CPU-only
        # monkeypatched test runs as well as on CUDA.
        if runtime is not None:
            runtime.clear()
        tensors.clear()
        _release_ocr_memory()


def _selected_ocr_inference_keys(config: dict) -> list[str]:
    _, keys = _ocr_model_keys(config)
    if isinstance(config.get("models"), dict):
        transfer = config["models"].get("transfer", [])
        return [transfer] if isinstance(transfer, str) else list(transfer)
    return keys


def run_ocr_inference(config: dict, event_callback=None) -> dict:
    inputs = config["inputs"]
    if "clean" in inputs and "adversarial" in inputs:
        image_paths = {
            "clean": Path(inputs["clean"]),
            "adversarial": Path(inputs["adversarial"]),
        }
    else:
        image_paths = {"image": Path(inputs["image"])}
    outputs = {}
    errors = []
    selected_keys = _selected_ocr_inference_keys(config)
    inference_config = dict(config)
    # A global prompt is only unambiguous for a direct, single-model inference
    # manifest.  Pipeline transfer always uses each model's native OCR prompt.
    inference_config["_allow_global_inference_prompt"] = (
        len(selected_keys) == 1
        and set(image_paths) == {"image"}
        and not isinstance(config.get("models"), dict)
        and not bool(config.get("_pipeline_transfer", False))
    )
    for model_key in selected_keys:
        try:
            outputs[model_key] = _run_one_ocr_inference(
                inference_config, model_key, image_paths, event_callback
            )
        except Exception as exc:
            error_record = {
                "model_key": model_key,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_record)
            if event_callback is not None:
                event_callback(
                    "error",
                    stage="inference",
                    model_key=model_key,
                    message=str(exc),
                    traceback=error_record["traceback"],
                )
    metrics = {"models_requested": len(selected_keys), "models_succeeded": len(outputs)}
    if "clean" in image_paths and "adversarial" in image_paths:
        metrics["textual_change_metrics_are_behavioral"] = True
        metrics["per_model_transcription_changes"] = {}
        for model_key, model_outputs in outputs.items():
            clean_text = model_outputs.get("clean", "")
            adversarial_text = model_outputs.get("adversarial", "")
            distance = levenshtein_distance(clean_text, adversarial_text)
            metrics["per_model_transcription_changes"][model_key] = {
                "exact_match": clean_text == adversarial_text,
                "character_edit_distance": distance,
                "character_edit_rate": distance / max(1, len(clean_text)),
            }
    return {"metrics": metrics, "raw_outputs": outputs, "artifacts": {}, "errors": errors}


def run_ocr_attack(config: dict, event_callback=None) -> dict:
    attack_key, _ = _ocr_model_keys(config)
    model_config = _ocr_config_for_model(config, attack_key)
    device = torch.device(model_config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    x_clean = load_image_tensor(Path(config["inputs"]["image"]), device)
    runtime = _load_ocr_runtime(model_config, x_clean, event_callback)
    if event_callback is not None:
        event_callback("stage_started", stage="clean_evaluation")
    clean_text = _generate_ocr_text(runtime, x_clean, int(config["generation"]["max_new_tokens"]))
    if not clean_text:
        raise RuntimeError("Clean OCR transcript was empty.")
    if is_encoder_decoder_family(runtime["state"]["model_family"]):
        teacher_forced_batch = build_encoder_decoder_teacher_forced_batch(
            runtime["processor"].tokenizer, runtime["prompt_model_inputs"], clean_text, device
        )
    else:
        teacher_forced_batch = build_teacher_forced_batch(
            runtime["tokenizer"] if runtime["tokenizer"] is not None else runtime["processor"].tokenizer,
            runtime["prompt_model_inputs"], clean_text, device
        )
    attack = config["attack"]
    if event_callback is not None:
        event_callback("stage_started", stage="optimization")
    x_adv, delta, final_loss, final_grad_inf = run_pgd(
        runtime["model"], teacher_forced_batch, runtime["state"], x_clean,
        epsilon=float(attack["epsilon"]), alpha=float(attack["alpha"]), steps=int(attack["steps"]),
        random_start=bool(attack["random_start"]), event_callback=event_callback,
    )
    adv_text = _generate_ocr_text(runtime, x_adv, int(config["generation"]["max_new_tokens"]))
    if event_callback is not None:
        event_callback("model_completed", model_key=attack_key, output_lengths={"clean": len(clean_text), "adversarial": len(adv_text)})
    configured_epsilon = float(attack["epsilon"])
    linf_delta = float(delta.abs().max().item())
    if linf_delta > configured_epsilon + 1e-6:
        raise RuntimeError(
            "Perturbation exceeds the configured L_inf bound: "
            f"{linf_delta:.6f} > {configured_epsilon:.6f}"
        )
    artifact_dir = Path(config["_artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    adv_path = artifact_dir / "adversarial.png"
    noise_path = artifact_dir / "perturbation.png"
    save_image_tensor(x_adv, adv_path)
    save_image_tensor(torch.clamp(delta.abs() * 10.0, 0.0, 1.0), noise_path)
    edit_distance = levenshtein_distance(clean_text, adv_text)
    metrics = {
        "exact_match": clean_text == adv_text,
        "character_edit_distance": edit_distance,
        "character_edit_rate": edit_distance / max(1, len(clean_text)),
        "source_attack_loss": final_loss,
        "gradient_inf": final_grad_inf,
        "gradient_norm": final_grad_inf,
        "perturbation_inf": linf_delta,
        "perturbation_norm": linf_delta,
        "textual_change_metrics_are_behavioral": True,
    }
    # The attack runtime is no longer needed once both transcripts and
    # artifacts have been produced.  Drop it before a pipeline loads transfer
    # models, which may otherwise contend with the attack model for VRAM.
    runtime.clear()
    _release_ocr_memory()
    return {
        "metrics": metrics,
        "raw_outputs": {"clean": clean_text, "adversarial": adv_text},
        "artifacts": {"adversarial_image": str(adv_path), "perturbation_visualization": str(noise_path)},
        "errors": [],
    }


def run_ocr_pipeline(config: dict, event_callback=None) -> dict:
    attack_result = run_ocr_attack(config, event_callback)
    # Keep this boundary explicit: attack_ocr may have held transient tensor
    # references until its frame was returned, so collect before transfer load.
    _release_ocr_memory()
    transfer_keys = _selected_ocr_inference_keys(config)
    clean_path = Path(config["inputs"]["image"])
    adv_path = Path(attack_result["artifacts"]["adversarial_image"])
    transfer_config = dict(config)
    transfer_config["inputs"] = {"clean": str(clean_path), "adversarial": str(adv_path)}
    transfer_config["models"] = transfer_keys
    transfer_config["_pipeline_transfer"] = True
    transfer = run_ocr_inference(transfer_config, event_callback)
    transfer_metrics = transfer["metrics"]
    attack_errors = attack_result.get("errors") or []
    transfer_errors = transfer.get("errors") or []
    return {
        "metrics": {
            **attack_result["metrics"],
            "attack": attack_result["metrics"],
            "transfer": transfer_metrics,
            # Retain the old flat count for consumers that used it, while the
            # nested transfer object preserves the complete per-model results.
            "transfer_models_succeeded": transfer_metrics["models_succeeded"],
        },
        "raw_outputs": {"attack": attack_result["raw_outputs"], "transfer": transfer["raw_outputs"]},
        "artifacts": {**(attack_result.get("artifacts") or {}), **(transfer.get("artifacts") or {})},
        "errors": [*attack_errors, *transfer_errors],
    }


def main() -> None:
    from experiment_runner import execute_run
    from workflow_contract import resolve_manifest

    print("Canonical CLI: /home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py run <CONFIG.json>")
    manifest = {
        "schema_version": 1,
        "name": f"legacy-ocr-{MODEL_KEY}-{IMG_IDX}",
        "workflow": "ocr_attack",
        "inputs": {"image": str(SOURCE_IMAGE_PATH)},
        "models": MODEL_KEY,
        "attack": {"epsilon": EPSILON, "alpha": ALPHA, "steps": STEPS, "random_start": RANDOM_START},
    }
    config = resolve_manifest(manifest)
    ok, _ = execute_run(config, "src/experiment_runner.py run <CONFIG.json>")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
