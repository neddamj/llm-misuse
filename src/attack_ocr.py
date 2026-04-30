import os
from pathlib import Path
from types import MethodType
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers.utils import logging as transformers_logging

from attacks.common import find_repo_root, load_image_tensor, project_delta, save_image_tensor
from attacks.prompting import build_chat_prompt_inputs, build_teacher_forced_batch, generate_greedy_text
from attacks.vision import build_qwen_vision_inputs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()

img_idx = 3
MODEL_KEY = "deepseek_ocr" # Options: "imgscope", "deepseek_ocr", "deepseek_ocr_2"
MODEL_CONFIGS = {
    "imgscope": {
        "model_name": "prithivMLmods/Imgscope-OCR-2B-0527",
        "model_family": "qwen",
        "ocr_prompt": "Read all text in the image and output only the extracted text.",
        "result_prefix": "imgscope_ocr_pgd",
    },
    "deepseek_ocr": {
        "model_name": "deepseek-ai/DeepSeek-OCR",
        "model_family": "deepseek",
        "ocr_prompt": "<image>\nFree OCR. ",
        "result_prefix": "deepseek_ocr_pgd",
        "base_size": 1024,
        "image_size": 640,
    },
    "deepseek_ocr_2": {
        "model_name": "deepseek-ai/DeepSeek-OCR-2",
        "model_family": "deepseek",
        "ocr_prompt": "<image>\nFree OCR. ",
        "result_prefix": "deepseek_ocr_2_pgd",
        "base_size": 1024,
        "image_size": 768,
    },
}
if MODEL_KEY not in MODEL_CONFIGS:
    supported_model_keys = ", ".join(sorted(MODEL_CONFIGS))
    raise ValueError(f"MODEL_KEY must be one of: {supported_model_keys}. Got {MODEL_KEY!r}.")

MODEL_CONFIG = MODEL_CONFIGS[MODEL_KEY]
MODEL_NAME = MODEL_CONFIG["model_name"]
MODEL_FAMILY = MODEL_CONFIG["model_family"]
SOURCE_IMAGE_PATH = Path(f"data/images/{img_idx}.png")
OCR_PROMPT = MODEL_CONFIG["ocr_prompt"]
EPSILON = 32 / 255
ALPHA = 2 / 255
STEPS = 100
MAX_NEW_TOKENS = 128
RANDOM_START = False
RESULT_PREFIX = MODEL_CONFIG["result_prefix"]

MODEL_INPUT_SIZE = 448

REPO_ROOT = find_repo_root(Path(__file__).resolve())
RESULTS_DIR = REPO_ROOT / "results"
SOURCE_IMAGE_PATH = REPO_ROOT / SOURCE_IMAGE_PATH
OUTPUT_ADV_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_adv.png"
OUTPUT_REPORT_PATH = RESULTS_DIR / f"{RESULT_PREFIX}_report.txt"

def ensure_deepseek_transformers_compat() -> None:
    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, "LlamaFlashAttention2"):
        llama_modeling.LlamaFlashAttention2 = llama_modeling.LlamaAttention


def patch_deepseek_forward(model):
    original_forward = model.model.forward

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
            inputs_embeds = self.get_input_embeddings()(input_ids).clone()

        sam_model = getattr(self, "sam_model", None)
        qwen2_model = getattr(self, "qwen2_model", None)
        vision_model = getattr(self, "vision_model", None)
        image_newline = getattr(self, "image_newline", None)

        if (
            sam_model is not None
            and images is not None
            and images_seq_mask is not None
            and input_ids is not None
            and (input_ids.shape[1] != 1 or self.training)
            and torch.sum(images[0][1]).item() != 0
        ):
            for idx, image in enumerate(images):
                patches = image[0]
                image_ori = image[1]

                if torch.sum(patches).item() != 0:
                    if qwen2_model is not None:
                        local_features = qwen2_model(sam_model(patches))
                        local_features = self.projector(local_features)
                        local_features = local_features.view(-1, local_features.shape[-1])
                        global_features = qwen2_model(sam_model(image_ori))
                        global_features = self.projector(global_features)
                        global_features = global_features.view(-1, global_features.shape[-1])
                        image_features = torch.cat(
                            [local_features, global_features, self.view_seperator[None, :]],
                            dim=0,
                        )
                    elif vision_model is not None:
                        if image_newline is None:
                            raise RuntimeError("DeepSeek OCR vision_model path requires image_newline.")
                        local_features_1 = sam_model(patches)
                        local_features_2 = vision_model(patches, local_features_1)
                        local_features = torch.cat(
                            (
                                local_features_2[:, 1:],
                                local_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        local_features = self.projector(local_features)

                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (
                                global_features_2[:, 1:],
                                global_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        global_features = self.projector(global_features)

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)
                        _, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2 ** 0.5)
                        width_crop_num = int(images_spatial_crop[idx][0].item())
                        height_crop_num = int(images_spatial_crop[idx][1].item())

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [global_features, image_newline[None, None, :].expand(h, 1, n_dim)],
                            dim=1,
                        )
                        global_features = global_features.view(-1, n_dim)

                        local_features = local_features.view(
                            height_crop_num,
                            width_crop_num,
                            h2,
                            w2,
                            n_dim2,
                        ).permute(0, 2, 1, 3, 4).reshape(
                            height_crop_num * h2,
                            width_crop_num * w2,
                            n_dim2,
                        )
                        local_features = torch.cat(
                            [
                                local_features,
                                image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2),
                            ],
                            dim=1,
                        )
                        local_features = local_features.view(-1, n_dim2)
                        image_features = torch.cat(
                            [local_features, global_features, self.view_seperator[None, :]],
                            dim=0,
                        )
                    else:
                        raise RuntimeError("DeepSeek OCR model does not expose a supported vision stack.")
                else:
                    if qwen2_model is not None:
                        global_features = qwen2_model(sam_model(image_ori))
                        global_features = self.projector(global_features)
                        global_features = global_features.view(-1, global_features.shape[-1])
                        image_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                    elif vision_model is not None:
                        if image_newline is None:
                            raise RuntimeError("DeepSeek OCR vision_model path requires image_newline.")
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (
                                global_features_2[:, 1:],
                                global_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        global_features = self.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)
                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [global_features, image_newline[None, None, :].expand(h, 1, n_dim)],
                            dim=1,
                        )
                        global_features = global_features.view(-1, n_dim)
                        image_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                    else:
                        raise RuntimeError("DeepSeek OCR model does not expose a supported vision stack.")

                mask = images_seq_mask[idx].unsqueeze(-1).to(device=inputs_embeds.device)
                inputs_embeds[idx].masked_scatter_(mask, image_features.to(inputs_embeds.dtype))

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
    return original_forward


def square_pad_and_normalize(
    image_tensor: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    height, width = image_tensor.shape[-2:]
    scale = image_size / max(height, width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_h = image_size - resized_h
    pad_w = image_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)
    return (x - 0.5) / 0.5


def square_resize_and_normalize(
    image_tensor: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    x = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (x - 0.5) / 0.5


def resolve_deepseek_layout(model) -> str:
    if getattr(model.model, "vision_model", None) is not None:
        return "ocr"
    if getattr(model.model, "qwen2_model", None) is not None:
        return "ocr2"
    raise RuntimeError("Unsupported DeepSeek OCR model layout.")


def build_deepseek_prompt_inputs(tokenizer, state: dict) -> dict[str, torch.Tensor]:
    module = state["deepseek_module"]
    prompt = state["prompt"]
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": ["dummy"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    rendered_prompt = module.format_messages(
        conversations=conversation,
        sft_format="plain",
        system_prompt="",
    )

    image_token = "<image>"
    image_token_id = 128815
    image_size = state["image_size"]
    patch_size = 16
    downsample_ratio = 4
    num_queries = (image_size // patch_size + downsample_ratio - 1) // downsample_ratio

    text_splits = rendered_prompt.split(image_token)
    if len(text_splits) != 2:
        raise RuntimeError("DeepSeek OCR prompt must contain exactly one <image> token.")

    tokenized_str = module.text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    images_seq_mask = [False] * len(tokenized_str)
    if state["deepseek_layout"] == "ocr":
        tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
        tokenized_image += [image_token_id]
    else:
        tokenized_image = ([image_token_id] * num_queries) * num_queries + [image_token_id]
    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = module.text_encode(tokenizer, text_splits[1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask
    input_ids = torch.tensor([tokenized_str], device=state["device"], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images_seq_mask": torch.tensor([images_seq_mask], device=state["device"], dtype=torch.bool),
        "images_spatial_crop": torch.tensor([[1, 1]], device=state["device"], dtype=torch.long),
    }


def build_deepseek_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict:
    if state["deepseek_layout"] == "ocr":
        images_ori = square_resize_and_normalize(image_tensor, state["image_size"])
    else:
        images_ori = square_pad_and_normalize(image_tensor, state["image_size"])
    images_ori = images_ori.to(dtype=state["dtype"]).unsqueeze(0)
    images_crop = torch.zeros(
        (1, 3, state["base_size"], state["base_size"]),
        device=image_tensor.device,
        dtype=state["dtype"],
    )
    return {
        "images": [(images_crop, images_ori)],
        "images_seq_mask": state["prompt_model_inputs"]["images_seq_mask"],
        "images_spatial_crop": state["prompt_model_inputs"]["images_spatial_crop"],
    }


def build_deepseek_teacher_forced_batch(tokenizer, prompt_model_inputs: dict, text: str, device: torch.device) -> dict:
    target_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (target_ids.shape[1] == 0 or target_ids[0, -1].item() != eos_token_id):
        eos_tensor = torch.tensor([[eos_token_id]], device=device, dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    prompt_input_ids = prompt_model_inputs["input_ids"]
    input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : prompt_input_ids.shape[1]] = -100
    prompt_mask = prompt_model_inputs["images_seq_mask"]
    target_mask = torch.zeros(target_ids.shape, device=device, dtype=torch.bool)
    return {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "labels": labels,
        "images_seq_mask": torch.cat([prompt_mask, target_mask], dim=1),
    }


def generate_deepseek_text(model, tokenizer, state: dict, image_tensor: torch.Tensor) -> str:
    prompt_model_inputs = state["prompt_model_inputs"]
    vision_inputs = build_deepseek_vision_inputs(state, image_tensor)
    generated = model.generate(
        input_ids=prompt_model_inputs["input_ids"],
        attention_mask=prompt_model_inputs["attention_mask"],
        images=vision_inputs["images"],
        images_seq_mask=vision_inputs["images_seq_mask"],
        images_spatial_crop=vision_inputs["images_spatial_crop"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    new_tokens = generated[:, prompt_model_inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    for stop_text in ("<｜end▁of▁sentence｜>", "<|end▁of▁sentence|>"):
        if text.endswith(stop_text):
            text = text[: -len(stop_text)]
    return text.strip()


def transcription_loss(
    model,
    teacher_forced_batch: dict[str, torch.Tensor],
    vision_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "images_seq_mask" in teacher_forced_batch:
        vision_inputs = dict(vision_inputs)
        vision_inputs["images_seq_mask"] = teacher_forced_batch["images_seq_mask"]
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

    # Guard against silent failures where quantization or preprocessing blocks gradient flow.
    saw_nonzero_grad = False
    last_loss = 0.0
    last_grad_inf = 0.0
    progress = tqdm(range(STEPS))
    for _ in progress:
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        if state["model_family"] == "deepseek":
            vision_inputs = build_deepseek_vision_inputs(state, x_adv.squeeze(0))
        else:
            vision_inputs = build_qwen_vision_inputs(state, x_adv.squeeze(0))
        loss = transcription_loss(model, teacher_forced_batch, vision_inputs)
        loss.backward()

        if delta.grad is None:
            raise RuntimeError("Expected PGD gradients on the perturbation tensor.")

        grad = delta.grad.detach()
        grad_inf = float(grad.abs().max().item())
        saw_nonzero_grad = saw_nonzero_grad or grad_inf > 0.0

        with torch.no_grad():
            # PGD ascent on transcription loss, then project back into the L_inf epsilon ball.
            delta.add_(ALPHA * grad.sign())
            project_delta(delta, x_clean, EPSILON)

        last_loss = float(loss.item())
        last_grad_inf = grad_inf
        progress.set_postfix(loss=f"{last_loss:.4f}", grad_inf=f"{last_grad_inf:.6f}")

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

    x_clean = load_image_tensor(SOURCE_IMAGE_PATH, device)

    if MODEL_FAMILY == "deepseek":
        ensure_deepseek_transformers_compat()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=model_dtype,
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        original_deepseek_forward = patch_deepseek_forward(model)
        model.model.forward = original_deepseek_forward
        state = {
            "device": device,
            "dtype": model_dtype,
            "model_family": MODEL_FAMILY,
            "prompt": OCR_PROMPT,
            "base_size": MODEL_CONFIG["base_size"],
            "image_size": MODEL_CONFIG["image_size"],
            "deepseek_layout": resolve_deepseek_layout(model),
            "deepseek_module": __import__(model.__class__.__module__, fromlist=[""]),
        }
        state["prompt_model_inputs"] = build_deepseek_prompt_inputs(tokenizer, state)
        with torch.no_grad():
            clean_text = generate_deepseek_text(model, tokenizer, state, x_clean.squeeze(0))
        if not clean_text:
            raise RuntimeError("Clean OCR transcript was empty.")
        teacher_forced_batch = build_deepseek_teacher_forced_batch(
            tokenizer,
            state["prompt_model_inputs"],
            clean_text,
            device,
        )
        patch_deepseek_forward(model)
    else:
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
            "model_family": MODEL_FAMILY,
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

    if MODEL_FAMILY == "deepseek":
        model.model.forward = original_deepseek_forward
        with torch.no_grad():
            adv_text = generate_deepseek_text(model, tokenizer, state, x_adv.squeeze(0))
    else:
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
