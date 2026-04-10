import math

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


SUPPORTED_MODEL_TYPES = {
    "gemma3": "gemma",
    "idefics3": "smolvlm",
    "qwen2_vl": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen3_vl": "qwen",
    "llava": "llava",
    "llava_next": "llava_next",
    "smolvlm": "smolvlm",
}


def resolve_model_family(requested_model_family: str, model_type: str) -> str:
    if requested_model_family not in {"auto", "gemma", "qwen", "llava", "llava_next", "smolvlm"}:
        raise ValueError(
            "MODEL_FAMILY must be one of: "
            "'auto', 'gemma', 'qwen', 'llava', 'llava_next', 'smolvlm'."
        )

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
    dtype: torch.dtype | None = None,
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
    if dtype is not None:
        pixel_values = pixel_values.to(dtype=dtype)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=device, dtype=torch.long)
    return pixel_values, image_grid_thw


def build_qwen_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    pixel_values, image_grid_thw = pack_for_qwen(
        image_tensor,
        model_input_size=state["model_input_size"],
        mean=state["mean"].view(3, 1, 1),
        std=state["std"].view(3, 1, 1),
        patch_size=state["patch_size"],
        temporal_patch_size=state["temporal_patch_size"],
        merge_size=state["merge_size"],
        device=state["device"],
        dtype=state.get("dtype"),
    )
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


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


def resize_bilinear(
    image_tensor: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    needs_batch_dim = image_tensor.ndim == 3
    x = image_tensor.unsqueeze(0) if needs_batch_dim else image_tensor
    x = F.interpolate(
        x,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    if needs_batch_dim:
        return x.squeeze(0)
    return x


def center_crop_or_pad(
    image_tensor: torch.Tensor,
    crop_size: tuple[int, int],
) -> torch.Tensor:
    needs_batch_dim = image_tensor.ndim == 3
    x = image_tensor.unsqueeze(0) if needs_batch_dim else image_tensor

    crop_h, crop_w = crop_size
    height, width = x.shape[-2:]

    if height < crop_h or width < crop_w:
        pad_h = max(0, crop_h - height)
        pad_w = max(0, crop_w - width)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        height, width = x.shape[-2:]

    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    x = x[:, :, top : top + crop_h, left : left + crop_w]

    if needs_batch_dim:
        return x.squeeze(0)
    return x


def maybe_rescale(image_tensor: torch.Tensor, rescale_factor: float) -> torch.Tensor:
    if torch.max(image_tensor) > 1.0:
        return image_tensor * rescale_factor
    return image_tensor


def resize_longest_edge(
    image_tensor: torch.Tensor,
    longest_edge: int,
) -> torch.Tensor:
    height, width = image_tensor.shape[-2:]
    aspect_ratio = width / height

    if width >= height:
        resized_width = longest_edge
        resized_height = int(resized_width / aspect_ratio)
        if resized_height % 2 != 0:
            resized_height += 1
    else:
        resized_height = longest_edge
        resized_width = int(resized_height * aspect_ratio)
        if resized_width % 2 != 0:
            resized_width += 1

    resized_height = max(resized_height, 1)
    resized_width = max(resized_width, 1)
    return resize_bilinear(image_tensor, (resized_height, resized_width))


def pack_for_llava_next(
    image_tensor: torch.Tensor,
    *,
    resize_size: tuple[int, int],
    crop_size: tuple[int, int],
    image_grid_pinpoints: list[tuple[int, int]],
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_height, original_width = image_tensor.shape[-2:]
    best_height, best_width = max(
        image_grid_pinpoints,
        key=lambda resolution: (
            min(
                int(original_width * min(resolution[1] / original_width, resolution[0] / original_height))
                * int(original_height * min(resolution[1] / original_width, resolution[0] / original_height)),
                original_width * original_height,
            ),
            -(
                resolution[0] * resolution[1]
                - min(
                    int(original_width * min(resolution[1] / original_width, resolution[0] / original_height))
                    * int(original_height * min(resolution[1] / original_width, resolution[0] / original_height)),
                    original_width * original_height,
                )
            ),
        ),
    )

    scale_w = best_width / original_width
    scale_h = best_height / original_height
    if scale_w < scale_h:
        resized_width = best_width
        resized_height = min(math.ceil(original_height * scale_w), best_height)
    else:
        resized_height = best_height
        resized_width = min(math.ceil(original_width * scale_h), best_width)

    padded_image = resize_bilinear(image_tensor, (resized_height, resized_width))
    pad_h = best_height - resized_height
    pad_w = best_width - resized_width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_image = F.pad(padded_image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    patch_size = crop_size[0]
    patches = padded_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image_tensor.shape[0], patch_size, patch_size)

    global_image = resize_bilinear(image_tensor, resize_size)
    global_image = center_crop_or_pad(global_image, crop_size).unsqueeze(0)

    local_images = resize_bilinear(patches, resize_size)
    local_images = center_crop_or_pad(local_images, crop_size)

    pixel_values = torch.cat([global_image, local_images], dim=0)
    pixel_values = maybe_rescale(pixel_values, rescale_factor)
    pixel_values = (pixel_values - mean) / std

    image_sizes = torch.tensor(
        [[original_height, original_width]],
        device=device,
        dtype=torch.long,
    )
    return pixel_values.unsqueeze(0), image_sizes


def pack_for_smolvlm(
    image_tensor: torch.Tensor,
    *,
    resize_longest_edge_value: int,
    max_image_size: int,
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = resize_longest_edge(image_tensor, resize_longest_edge_value)
    x = resize_bilinear(x, (max_image_size, max_image_size))
    x = maybe_rescale(x, rescale_factor)
    x = (x - mean) / std

    pixel_values = x.unsqueeze(0).unsqueeze(0)
    pixel_attention_mask = torch.ones(
        (1, 1, max_image_size, max_image_size),
        device=image_tensor.device,
        dtype=torch.bool,
    )
    return pixel_values, pixel_attention_mask


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


def build_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    vision_state = state["vision_state"]
    if state["model_family"] == "qwen":
        return build_qwen_vision_inputs(vision_state, image_tensor)

    if state["model_family"] == "gemma":
        pixel_values = pack_for_gemma(
            image_tensor,
            size=vision_state["size"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
        )
        return {"pixel_values": pixel_values}

    if state["model_family"] == "llava_next":
        pixel_values, image_sizes = pack_for_llava_next(
            image_tensor,
            resize_size=vision_state["resize_size"],
            crop_size=vision_state["crop_size"],
            image_grid_pinpoints=vision_state["image_grid_pinpoints"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
            device=state["device"],
        )
        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }

    if state["model_family"] == "smolvlm":
        pixel_values, pixel_attention_mask = pack_for_smolvlm(
            image_tensor,
            resize_longest_edge_value=vision_state["resize_longest_edge"],
            max_image_size=vision_state["max_image_size"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(3, 1, 1),
            std=vision_state["std"].view(3, 1, 1),
        )
        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }

    pixel_values = pack_for_llava(
        image_tensor,
        shortest_edge=vision_state["shortest_edge"],
        crop_size=vision_state["crop_size"],
        mean=vision_state["mean"].view(1, 3, 1, 1),
        std=vision_state["std"].view(1, 3, 1, 1),
    )
    return {"pixel_values": pixel_values}


def sample_camera_transform(
    image_tensor: torch.Tensor,
    *,
    rotation_degrees: float,
    perspective_distortion: float,
    crop_scale: tuple[float, float],
    crop_ratio: tuple[float, float],
    color_jitter_brightness: float,
    color_jitter_contrast: float,
    color_jitter_saturation: float,
    gaussian_noise_std: float,
) -> torch.Tensor:
    if image_tensor.ndim != 3:
        raise ValueError("Expected image_tensor with shape (C, H, W).")

    channels, height, width = image_tensor.shape
    fill = [1.0] * channels

    startpoints, endpoints = transforms.RandomPerspective.get_params(
        width=width,
        height=height,
        distortion_scale=perspective_distortion,
    )
    x = TF.perspective(
        image_tensor,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BILINEAR,
        fill=fill,
    )

    top, left, crop_height, crop_width = transforms.RandomResizedCrop.get_params(
        x,
        scale=crop_scale,
        ratio=crop_ratio,
    )
    x = TF.resized_crop(
        x,
        top,
        left,
        crop_height,
        crop_width,
        size=[height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    angle = float(torch.empty(1).uniform_(-rotation_degrees, rotation_degrees).item())
    x = TF.rotate(
        x,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=fill,
    )

    brightness_factor = 1.0 + float(
        torch.empty(1).uniform_(-color_jitter_brightness, color_jitter_brightness).item()
    )
    contrast_factor = 1.0 + float(
        torch.empty(1).uniform_(-color_jitter_contrast, color_jitter_contrast).item()
    )
    saturation_factor = 1.0 + float(
        torch.empty(1).uniform_(-color_jitter_saturation, color_jitter_saturation).item()
    )

    x = TF.adjust_brightness(x, brightness_factor)
    x = TF.adjust_contrast(x, contrast_factor)
    x = TF.adjust_saturation(x, saturation_factor)
    x = x + torch.randn_like(x) * gaussian_noise_std
    return torch.clamp(x, 0.0, 1.0)
