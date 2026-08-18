"""Canonical CLI for reproducible OCR and VLM experiments."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from workflow_contract import (
    EventLog,
    REPO_ROOT,
    SCHEMA_VERSION,
    VLM_ATTACK_MODELS,
    VLM_INFERENCE_MODELS,
    OCR_MODELS,
    WORKFLOWS,
    batch_entries,
    canonical_json,
    check_cuda_devices,
    check_interpreter,
    config_hash,
    create_run_dir,
    deterministic_summary,
    example_manifest,
    expected_interpreter,
    load_json,
    relative_artifact_paths,
    resolve_manifest,
    runtime_metadata,
    utc_now,
    write_status,
)


def _set_value(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"--set expects path=value, got {expression!r}.")
    path, raw_value = expression.split("=", 1)
    if not path or any(not part for part in path.split(".")):
        raise ValueError(f"--set path must be a dotted, non-empty path: {path!r}.")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _load_resolved(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    raw = load_json(path)
    for expression in overrides or []:
        _set_value(raw, expression)
    return resolve_manifest(raw)


def _workflow_function(workflow: str):
    # These imports intentionally stay here: listing, validating, and summarizing
    # must not import the optional OCR/VLM model stacks.
    if workflow.startswith("ocr"):
        from ocr_workflows import workflow_function
    else:
        from vlm_workflows import workflow_function
    return workflow_function(workflow)


def execute_run(config: dict[str, Any], command_hint: str) -> tuple[bool, Path]:
    check_interpreter(config["workflow"], command_hint)
    run_dir = create_run_dir(config)
    event_log = EventLog(run_dir)
    saved_config = copy.deepcopy(config)
    (run_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    status = {
        "run_id": run_dir.name,
        "workflow": config["workflow"],
        "name": config["name"],
        "config_hash": config_hash(config),
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "error": None,
        **runtime_metadata(config),
    }
    write_status(run_dir, status)
    event_log("run_started", workflow=config["workflow"], run_id=run_dir.name)
    runtime_config = copy.deepcopy(config)
    runtime_config["_run_dir"] = str(run_dir)
    runtime_config["_artifact_dir"] = str(run_dir / "artifacts")

    try:
        check_cuda_devices(config)
        result = _workflow_function(config["workflow"])(runtime_config, event_log)
        if not isinstance(result, dict):
            raise RuntimeError("Workflow returned a non-object result.")
        result = relative_artifact_paths(result, run_dir)
        errors = result.get("errors") or []
        final_status = "completed_with_errors" if errors else "completed"
        status["status"] = final_status
        event_log("run_completed", status=final_status, error_count=len(errors))
    except Exception as exc:
        result = {
            "metrics": {},
            "raw_outputs": {},
            "artifacts": {},
            "errors": [{"stage": "run", "type": type(exc).__name__, "message": str(exc)}],
        }
        status["status"] = "failed"
        status["error"] = {"type": type(exc).__name__, "message": str(exc)}
        event_log("run_failed", error_type=type(exc).__name__, message=str(exc))
    result_path = run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(
        deterministic_summary(config, result, status["status"]),
        encoding="utf-8",
    )
    status["finished_at"] = utc_now()
    write_status(run_dir, status)
    print(f"Run directory: {run_dir}")
    return status["status"] in {"completed", "completed_with_errors"}, run_dir


def _list_models(args: argparse.Namespace | None = None) -> int:
    payload = {
        "ocr": {key: OCR_MODELS[key] for key in sorted(OCR_MODELS)},
        "vlm_attack": {key: VLM_ATTACK_MODELS[key] for key in sorted(VLM_ATTACK_MODELS)},
        "vlm_inference": {key: VLM_INFERENCE_MODELS[key] for key in sorted(VLM_INFERENCE_MODELS)},
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _example(args: argparse.Namespace) -> int:
    manifest = example_manifest(args.workflow)
    rendered = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote example manifest to {output.resolve()}")
    else:
        print(rendered, end="")
    return 0


def _validate(args: argparse.Namespace) -> int:
    resolved = _load_resolved(Path(args.config).expanduser().resolve())
    print(json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = _load_resolved(config_path, args.set_values)
        ok, _ = execute_run(config, f"src/experiment_runner.py run {config_path}")
        return 0 if ok else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _batch(args: argparse.Namespace) -> int:
    batch_path = Path(args.batch).expanduser().resolve()
    raw = load_json(batch_path)
    entries = batch_entries(raw, batch_path)
    failures = 0
    completed: list[str] = []
    for index, entry in enumerate(entries):
        config_path = Path(entry["config"])
        try:
            config = _load_resolved(config_path, [f"{key}={json.dumps(value)}" for key, value in entry.get("set", {}).items()])
            ok, run_dir = execute_run(
                config,
                f"src/experiment_runner.py run {config_path}",
            )
            completed.append(str(run_dir))
            if not ok:
                failures += 1
                if args.fail_fast:
                    break
        except Exception as exc:
            failures += 1
            print(f"Batch entry {index} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    result = {"status": "failed" if failures else "completed", "failures": failures, "runs": completed}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


def _summarize(args: argparse.Namespace) -> int:
    runs = []
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path).expanduser().resolve()
        status = load_json(run_dir / "status.json")
        results = load_json(run_dir / "results.json")
        runs.append(
            {
                "run_id": status.get("run_id", run_dir.name),
                "workflow": status.get("workflow"),
                "status": status.get("status"),
                "metrics": results.get("metrics", {}),
                "path": str(run_dir),
            }
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(str(run["workflow"]), []).append(run)
    compatible_groups = []
    for workflow, group in sorted(groups.items()):
        common = None
        for run in group:
            keys = set(run["metrics"])
            common = keys if common is None else common & keys
        compatible_groups.append({"workflow": workflow, "metric_keys": sorted(common or []), "run_ids": [run["run_id"] for run in group]})
    payload = {"runs": runs, "compatible_groups": compatible_groups}
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    lines = ["# Experiment summary", "", "| Run | Workflow | Status | Comparable metrics |", "|---|---|---|---|"]
    for run in runs:
        metric_names = ", ".join(sorted(run["metrics"])) or "—"
        lines.append(f"| `{run['run_id']}` | `{run['workflow']}` | `{run['status']}` | {metric_names} |")
    lines.extend(["", "Runs are compared only within the same workflow; no universal score or external judge is used."])
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List explicit OCR and VLM model selections.")

    example = subparsers.add_parser("example", help="Print or write an example manifest.")
    example.add_argument("--workflow", required=True, choices=WORKFLOWS)
    example.add_argument("--output")

    validate = subparsers.add_parser("validate", help="Resolve and validate a manifest without loading weights.")
    validate.add_argument("config")

    run = subparsers.add_parser("run", help="Validate and execute one manifest.")
    run.add_argument("config")
    run.add_argument("--set", dest="set_values", action="append", default=[])

    batch = subparsers.add_parser("batch", help="Execute an explicit sequential batch.")
    batch.add_argument("batch")
    batch.add_argument("--fail-fast", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Compare compatible saved run metrics.")
    summarize.add_argument("run_dirs", nargs="+")
    summarize.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "list-models": _list_models,
        "example": _example,
        "validate": _validate,
        "run": _run,
        "batch": _batch,
        "summarize": _summarize,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
