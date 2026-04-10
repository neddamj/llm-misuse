from pathlib import Path

import numpy as np
import torch
from PIL import Image


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo root from the current working directory. "
        "Launch the script from this repository or one of its subdirectories."
    )


def load_image_tensor(
    image_path: str | Path,
    device: torch.device,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if image_size is not None:
        image = image.resize(image_size, Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def save_image_tensor(image_tensor: torch.Tensor, output_path: Path) -> None:
    image = image_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    array = (
        image.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array).save(output_path)


def project_delta(delta: torch.Tensor, x_clean: torch.Tensor, epsilon: float) -> None:
    with torch.no_grad():
        delta.clamp_(-epsilon, epsilon)
        delta.copy_(torch.clamp(x_clean + delta, 0.0, 1.0) - x_clean)


def canonicalize_cuda_device(device_name: str) -> str:
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None:
        raise ValueError(f"Expected an explicit CUDA device like 'cuda:0', got {device_name!r}.")
    return f"cuda:{device.index}"


def summarize_loss_values(
    losses_by_key: dict[str, float],
    *,
    higher_is_worse: bool = False,
) -> tuple[str, float, float]:
    if not losses_by_key:
        raise ValueError("Expected at least one loss value.")

    first_key = next(iter(losses_by_key))
    worst_key = first_key
    worst_loss = losses_by_key[first_key]
    total_loss = 0.0
    for key, loss in losses_by_key.items():
        total_loss += loss
        if (higher_is_worse and loss > worst_loss) or (not higher_is_worse and loss < worst_loss):
            worst_key = key
            worst_loss = loss

    return worst_key, worst_loss, total_loss / len(losses_by_key)
