"""Lazy OCR workflow dispatchers.

The model-heavy module is imported only when an OCR workflow is executed.
"""

from __future__ import annotations

from typing import Any, Callable


def workflow_function(workflow: str) -> Callable[[dict[str, Any], Callable[..., None]], dict[str, Any]]:
    from attack_ocr import run_ocr_attack, run_ocr_inference, run_ocr_pipeline

    functions = {
        "ocr_attack": run_ocr_attack,
        "ocr_inference": run_ocr_inference,
        "ocr_pipeline": run_ocr_pipeline,
    }
    try:
        return functions[workflow]
    except KeyError as exc:
        raise ValueError(f"Unsupported OCR workflow: {workflow!r}") from exc
