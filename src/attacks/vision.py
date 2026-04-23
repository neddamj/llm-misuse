import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


SUPPORTED_MODEL_TYPES = {
    "aya_vision": "aya_vision",
    "gemma3": "gemma",
    "instructblip": "instructblip",
    "jvlm": "jina_vlm",
    "idefics3": "smolvlm",
    "lfm2_vl": "lfm2_vl",
    "qwen2_vl": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen3_vl": "qwen",
    "llava": "llava",
    "llava_next": "llava_next",
    "smolvlm": "smolvlm",
}


def resolve_model_family(requested_model_family: str, model_type: str) -> str:
    if requested_model_family not in {
        "auto",
        "aya_vision",
        "gemma",
        "instructblip",
        "jina_vlm",
        "lfm2_vl",
        "qwen",
        "llava",
        "llava_next",
        "smolvlm",
    }:
        raise ValueError(
            "MODEL_FAMILY must be one of: "
            "'auto', 'aya_vision', 'gemma', 'instructblip', 'jina_vlm', 'lfm2_vl', "
            "'qwen', 'llava', 'llava_next', 'smolvlm'."
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


@lru_cache(maxsize=None)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> tuple[tuple[int, int], ...]:
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return tuple(aspect_ratios)


def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    possible_resolutions = get_all_supported_aspect_ratios(max_image_tiles)
    possible_resolutions = sorted(possible_resolutions, key=lambda ratio: ratio[0] * ratio[1])
    image_height, image_width = original_image_size
    patch_size_height, _ = target_tile_size

    best_grid = possible_resolutions[0]
    best_scale = None
    for grid_width, grid_height in possible_resolutions:
        if grid_width * grid_height < min_image_tiles:
            continue
        target_width = grid_width * patch_size_height
        target_height = grid_height * patch_size_height
        scale = min(target_width / image_width, target_height / image_height)
        if best_scale is None:
            best_scale = scale
            best_grid = (grid_width, grid_height)
            continue
        if best_scale < 1 and scale > best_scale:
            best_scale = scale
            best_grid = (grid_width, grid_height)
            continue
        if best_scale >= 1:
            adjusted_scale = scale if scale >= 1 else float("inf")
            adjusted_best_scale = best_scale if best_scale >= 1 else float("inf")
            if adjusted_scale < adjusted_best_scale:
                best_scale = scale
                best_grid = (grid_width, grid_height)
    return best_grid


def pack_for_aya_vision(
    image_tensor: torch.Tensor,
    *,
    tile_size: tuple[int, int],
    min_patches: int,
    max_patches: int,
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    use_thumbnail: bool = True,
) -> torch.Tensor:
    patch_height, patch_width = tile_size
    original_height, original_width = image_tensor.shape[-2:]
    num_columns, num_rows = get_optimal_tiled_canvas(
        (original_height, original_width),
        tile_size,
        min_patches,
        max_patches,
    )
    target_width = patch_width * num_columns
    target_height = patch_height * num_rows
    resized_image = resize_bilinear(image_tensor, (target_height, target_width))

    processed_images = []
    for row in range(num_rows):
        for column in range(num_columns):
            top = row * patch_height
            left = column * patch_width
            processed_images.append(resized_image[:, top : top + patch_height, left : left + patch_width])

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(resize_bilinear(image_tensor, tile_size))

    pixel_values = torch.stack(processed_images, dim=0)
    pixel_values = maybe_rescale(pixel_values, rescale_factor)
    return (pixel_values - mean) / std


def resize_bilinear(
    image_tensor: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if image_tensor.shape[-2:] == size:
        return image_tensor

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
    if image_tensor.shape[-2:] == crop_size:
        return image_tensor

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


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def convert_image_to_patches_channel_first(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, C, H, W).")
    batch_size, num_channels, image_height, image_width = images.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = images.reshape(
        batch_size,
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
    return patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)


def pad_patches(
    patches: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_length = patches.shape[0]
    padding_length = target_length - current_length
    patch_mask = torch.ones((target_length,), device=patches.device, dtype=torch.bool)
    if padding_length <= 0:
        return patches, patch_mask

    padded = torch.zeros(
        (target_length, patches.shape[1]),
        device=patches.device,
        dtype=patches.dtype,
    )
    padded[:current_length] = patches
    patch_mask[current_length:] = False
    return padded, patch_mask


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


def smart_resize_lfm2(
    height: int,
    width: int,
    *,
    downsample_factor: int,
    min_image_tokens: int,
    max_image_tokens: int,
    encoder_patch_size: int,
) -> tuple[int, int]:
    total_factor = encoder_patch_size * downsample_factor
    smart_resize_min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
    smart_resize_max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

    h_bar = max(total_factor, round_by_factor(height, total_factor))
    w_bar = max(total_factor, round_by_factor(width, total_factor))

    if h_bar * w_bar > smart_resize_max_pixels:
        beta = math.sqrt((height * width) / smart_resize_max_pixels)
        h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
        w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
    elif h_bar * w_bar < smart_resize_min_pixels:
        beta = math.sqrt(smart_resize_min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    return w_bar, h_bar


def is_image_too_large_lfm2(
    height: int,
    width: int,
    *,
    max_image_tokens: int,
    encoder_patch_size: int,
    downsample_factor: int,
    max_pixels_tolerance: float,
) -> bool:
    total_factor = encoder_patch_size * downsample_factor
    h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
    w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
    return (
        h_bar * w_bar
        > max_image_tokens * encoder_patch_size**2 * downsample_factor**2 * max_pixels_tolerance
    )


def pack_for_lfm2_vl(
    image_tensor: torch.Tensor,
    *,
    downsample_factor: int,
    do_image_splitting: bool,
    min_tiles: int,
    max_tiles: int,
    use_thumbnail: bool,
    min_image_tokens: int,
    max_image_tokens: int,
    encoder_patch_size: int,
    tile_size: int,
    max_pixels_tolerance: float,
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not do_image_splitting:
        min_tiles = 1
        max_tiles = 1

    max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
    tile_size_patches = (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0
    max_num_patches = max(max_thumbnail_image_patches, tile_size_patches)

    height, width = image_tensor.shape[-2:]
    is_large = is_image_too_large_lfm2(
        height,
        width,
        max_image_tokens=max_image_tokens,
        encoder_patch_size=encoder_patch_size,
        downsample_factor=downsample_factor,
        max_pixels_tolerance=max_pixels_tolerance,
    )
    new_width, new_height = smart_resize_lfm2(
        height,
        width,
        downsample_factor=downsample_factor,
        min_image_tokens=min_image_tokens,
        max_image_tokens=max_image_tokens,
        encoder_patch_size=encoder_patch_size,
    )

    if is_large and min_tiles != max_tiles:
        grid_width, grid_height = get_optimal_tiled_canvas(
            (height, width),
            (tile_size, tile_size),
            min_tiles,
            max_tiles,
        )
        target_width = tile_size * grid_width
        target_height = tile_size * grid_height
        resized_image = resize_bilinear(image_tensor, (target_height, target_width))
        processed_images = []
        for row in range(grid_height):
            for column in range(grid_width):
                top = row * tile_size
                left = column * tile_size
                processed_images.append(resized_image[:, top : top + tile_size, left : left + tile_size])
        if use_thumbnail and grid_width * grid_height != 1:
            processed_images.append(resize_bilinear(image_tensor, (new_height, new_width)))
    else:
        processed_images = [resize_bilinear(image_tensor, (new_height, new_width))]

    pixel_values_list = []
    pixel_attention_masks = []
    spatial_shapes = []
    for processed_image in processed_images:
        normalized_image = maybe_rescale(processed_image, rescale_factor)
        normalized_image = (normalized_image - mean.view(3, 1, 1)) / std.view(3, 1, 1)

        num_patches_height = normalized_image.shape[-2] // encoder_patch_size
        num_patches_width = normalized_image.shape[-1] // encoder_patch_size
        patches = convert_image_to_patches_channel_first(normalized_image.unsqueeze(0), encoder_patch_size).squeeze(0)
        patches, patch_mask = pad_patches(patches, max_num_patches)

        pixel_values_list.append(patches)
        pixel_attention_masks.append(patch_mask)
        spatial_shapes.append([num_patches_height, num_patches_width])

    pixel_values = torch.stack(pixel_values_list, dim=0)
    pixel_attention_mask = torch.stack(pixel_attention_masks, dim=0)
    spatial_shapes_tensor = torch.tensor(spatial_shapes, device=image_tensor.device, dtype=torch.long)
    return pixel_values, pixel_attention_mask, spatial_shapes_tensor


def smart_resize_jina(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_absolute_aspect_ratio: int = 200,
) -> tuple[int, int]:
    abs_aspect_ratio = max(height, width) / min(height, width)
    if abs_aspect_ratio > max_absolute_aspect_ratio:
        raise ValueError(
            f"Absolute aspect ratio must be < {max_absolute_aspect_ratio}, got {abs_aspect_ratio}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


@lru_cache(maxsize=None)
def get_molmo_tilings(max_num_crops: int) -> tuple[tuple[int, int], ...]:
    tilings = []
    for rows in range(1, max_num_crops + 1):
        for columns in range(1, max_num_crops + 1):
            if rows * columns <= max_num_crops:
                tilings.append((rows, columns))
    tilings.sort(key=lambda tiling: (tiling[0] * tiling[1], tiling[0]))
    return tuple(tilings)


def molmo_select_tiling(height: int, width: int, patch_size: int, max_num_crops: int) -> tuple[int, int]:
    tilings = get_molmo_tilings(max_num_crops)
    best_tiling = tilings[0]
    best_scale = None
    for rows, columns in tilings:
        target_height = rows * patch_size
        target_width = columns * patch_size
        scale = min(target_height / height, target_width / width)
        if best_scale is None:
            best_scale = scale
            best_tiling = (rows, columns)
            continue
        if best_scale < 1 and scale > best_scale:
            best_scale = scale
            best_tiling = (rows, columns)
            continue
        if best_scale >= 1:
            adjusted_scale = scale if scale >= 1 else float("inf")
            adjusted_best_scale = best_scale if best_scale >= 1 else float("inf")
            if adjusted_scale < adjusted_best_scale:
                best_scale = scale
                best_tiling = (rows, columns)
    return best_tiling


def molmo_get_patches_from_tiling(
    num_tiles: int,
    pooling_size: int,
    crop_patches: int,
    crop_window_patches: int,
    left_margin: int,
    right_margin: int,
) -> int:
    if num_tiles > 1:
        left_crop_window_patches = (
            (crop_window_patches + left_margin + pooling_size - 1) // pooling_size * pooling_size
        )
        middle_crop_window_patches = (
            (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
        )
        right_crop_window_patches = (
            (crop_window_patches + right_margin + pooling_size - 1) // pooling_size * pooling_size
        )
        return (
            left_crop_window_patches
            + (num_tiles - 2) * middle_crop_window_patches
            + right_crop_window_patches
        )
    return (crop_patches + pooling_size - 1) // pooling_size * pooling_size


def pack_for_jina_vlm(
    image_tensor: torch.Tensor,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int,
    max_crops: int,
    base_input_size: tuple[int, int],
    overlap_margins: tuple[int, int],
    pooling_w: int,
    pooling_h: int,
    token_length_w: int,
    token_length_h: int,
    use_column_tokens: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_tensor.shape[-2:]
    resized_height, resized_width = smart_resize_jina(
        height,
        width,
        factor=patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image_tensor = resize_bilinear(image_tensor, (resized_height, resized_width))

    left_margin, right_margin = overlap_margins
    total_margin_pixels = patch_size * (right_margin + left_margin)
    crop_patches = base_input_size[0] // patch_size
    crop_window_patches = crop_patches - (right_margin + left_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling_rows, tiling_columns = molmo_select_tiling(
        resized_height - total_margin_pixels,
        resized_width - total_margin_pixels,
        crop_window_size,
        max_crops,
    )
    src = resize_bilinear(
        image_tensor,
        (
            tiling_rows * crop_window_size + total_margin_pixels,
            tiling_columns * crop_window_size + total_margin_pixels,
        ),
    )
    src = (src - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    img_mask = torch.ones(src.shape[-2:], device=src.device, dtype=src.dtype)

    image_base_patch_w = base_input_size[1] // patch_size
    image_base_patch_h = base_input_size[0] // patch_size
    crop_size = base_input_size[0]

    patches_arr = []
    mask_arr = []
    for row in range(tiling_rows):
        y0 = row * crop_window_size
        for column in range(tiling_columns):
            x0 = column * crop_window_size
            patches_arr.append(src[:, y0 : y0 + crop_size, x0 : x0 + crop_size])
            mask_arr.append(img_mask[y0 : y0 + crop_size, x0 : x0 + crop_size])

    local_patches = torch.stack(patches_arr, dim=0)
    local_masks = torch.stack(mask_arr, dim=0)
    local_patches = convert_image_to_patches_channel_first(local_patches, patch_size)
    local_masks = convert_image_to_patches_channel_first(local_masks.unsqueeze(1), patch_size).mean(dim=-1)

    global_image = resize_bilinear(image_tensor, base_input_size)
    global_image = (global_image - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    global_patches = convert_image_to_patches_channel_first(global_image.unsqueeze(0), patch_size)
    image_patches = torch.cat([global_patches, local_patches], dim=0)

    image_masks = F.pad(local_masks, (0, 0, 0, 1), value=-1.0)
    return image_patches.unsqueeze(0), image_masks.unsqueeze(0)


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


def pack_for_instructblip(
    image_tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    rescale_factor: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x = resize_bilinear(image_tensor, size).unsqueeze(0)
    x = maybe_rescale(x, rescale_factor)
    return (x - mean) / std


def build_vision_inputs(state: dict, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    vision_state = state["vision_state"]
    if state["model_family"] == "qwen":
        return build_qwen_vision_inputs(vision_state, image_tensor)

    if state["model_family"] == "aya_vision":
        pixel_values = pack_for_aya_vision(
            image_tensor,
            tile_size=vision_state["tile_size"],
            min_patches=vision_state["min_patches"],
            max_patches=vision_state["max_patches"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
            use_thumbnail=vision_state["use_thumbnail"],
        )
        return {"pixel_values": pixel_values}

    if state["model_family"] == "gemma":
        pixel_values = pack_for_gemma(
            image_tensor,
            size=vision_state["size"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
        )
        return {"pixel_values": pixel_values}

    if state["model_family"] == "instructblip":
        pixel_values = pack_for_instructblip(
            image_tensor,
            size=vision_state["size"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"].view(1, 3, 1, 1),
            std=vision_state["std"].view(1, 3, 1, 1),
        )
        return {"pixel_values": pixel_values}

    if state["model_family"] == "jina_vlm":
        image_patches, image_masks = pack_for_jina_vlm(
            image_tensor,
            min_pixels=vision_state["min_pixels"],
            max_pixels=vision_state["max_pixels"],
            patch_size=vision_state["patch_size"],
            max_crops=vision_state["max_crops"],
            base_input_size=vision_state["base_input_size"],
            overlap_margins=vision_state["overlap_margins"],
            pooling_w=vision_state["pooling_w"],
            pooling_h=vision_state["pooling_h"],
            token_length_w=vision_state["token_length_w"],
            token_length_h=vision_state["token_length_h"],
            use_column_tokens=vision_state["use_column_tokens"],
            mean=vision_state["mean"],
            std=vision_state["std"],
        )
        return {
            "image_patches": image_patches,
            "image_masks": image_masks,
        }

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

    if state["model_family"] == "lfm2_vl":
        pixel_values, pixel_attention_mask, spatial_shapes = pack_for_lfm2_vl(
            image_tensor,
            downsample_factor=vision_state["downsample_factor"],
            do_image_splitting=vision_state["do_image_splitting"],
            min_tiles=vision_state["min_tiles"],
            max_tiles=vision_state["max_tiles"],
            use_thumbnail=vision_state["use_thumbnail"],
            min_image_tokens=vision_state["min_image_tokens"],
            max_image_tokens=vision_state["max_image_tokens"],
            encoder_patch_size=vision_state["encoder_patch_size"],
            tile_size=vision_state["tile_size"],
            max_pixels_tolerance=vision_state["max_pixels_tolerance"],
            rescale_factor=vision_state["rescale_factor"],
            mean=vision_state["mean"],
            std=vision_state["std"],
        )
        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
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

    if (
        rotation_degrees == 0
        and perspective_distortion == 0
        and crop_scale == (1.0, 1.0)
        and crop_ratio == (1.0, 1.0)
        and color_jitter_brightness == 0
        and color_jitter_contrast == 0
        and color_jitter_saturation == 0
        and gaussian_noise_std == 0
    ):
        return image_tensor

    channels, height, width = image_tensor.shape
    fill = [1.0] * channels

    x = image_tensor

    if perspective_distortion != 0:
        startpoints, endpoints = transforms.RandomPerspective.get_params(
            width=width,
            height=height,
            distortion_scale=perspective_distortion,
        )
        x = TF.perspective(
            x,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )

    if crop_scale != (1.0, 1.0) or crop_ratio != (1.0, 1.0):
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

    if rotation_degrees != 0:
        angle = float(torch.empty(1).uniform_(-rotation_degrees, rotation_degrees).item())
        x = TF.rotate(
            x,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )

    if color_jitter_brightness != 0:
        brightness_factor = 1.0 + float(
            torch.empty(1).uniform_(-color_jitter_brightness, color_jitter_brightness).item()
        )
        x = TF.adjust_brightness(x, brightness_factor)

    if color_jitter_contrast != 0:
        contrast_factor = 1.0 + float(
            torch.empty(1).uniform_(-color_jitter_contrast, color_jitter_contrast).item()
        )
        x = TF.adjust_contrast(x, contrast_factor)

    if color_jitter_saturation != 0:
        saturation_factor = 1.0 + float(
            torch.empty(1).uniform_(-color_jitter_saturation, color_jitter_saturation).item()
        )
        x = TF.adjust_saturation(x, saturation_factor)

    if gaussian_noise_std != 0:
        x = x + torch.randn_like(x) * gaussian_noise_std
    return torch.clamp(x, 0.0, 1.0)
