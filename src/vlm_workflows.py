"""Lazy VLM workflow dispatchers.

The model-heavy module is imported only when a VLM workflow is executed.
"""

from __future__ import annotations

from typing import Any, Callable


def workflow_function(workflow: str) -> Callable[[dict[str, Any], Callable[..., None]], dict[str, Any]]:
    from attack_vlm import run_vlm_attack, run_vlm_inference, run_vlm_pipeline

    functions = {
        "vlm_attack": run_vlm_attack,
        "vlm_inference": run_vlm_inference,
        "vlm_pipeline": run_vlm_pipeline,
    }
    try:
        return functions[workflow]
    except KeyError as exc:
        raise ValueError(f"Unsupported VLM workflow: {workflow!r}") from exc
