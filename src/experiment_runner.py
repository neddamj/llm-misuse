"""Canonical CLI for reproducible OCR and VLM experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import traceback
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
    _flatten_scalar_metrics,
    load_json,
    relative_artifact_paths,
    resolve_manifest,
    runtime_metadata,
    utc_now,
    write_status,
)


class _TerminalTee:
    """Write terminal output to both the original stream and a run log."""

    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


class _TerminalTeeContext:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.handle = self.path.open("a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = _TerminalTee(self.stdout, self.handle)
        sys.stderr = _TerminalTee(self.stderr, self.handle)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.handle.close()
        return False


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
    runtime_config = copy.deepcopy(config)
    runtime_config["_run_dir"] = str(run_dir)
    runtime_config["_artifact_dir"] = str(run_dir / "artifacts")

    with _TerminalTeeContext(run_dir / "run.log"):
        event_log("run_started", workflow=config["workflow"], run_id=run_dir.name)
        try:
            check_cuda_devices(config)
            result = _workflow_function(config["workflow"])(runtime_config, event_log)
            if not isinstance(result, dict):
                raise RuntimeError("Workflow returned a non-object result.")
            result = relative_artifact_paths(result, run_dir)
            errors = result.get("errors") or []
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            requested = metrics.get("models_requested")
            succeeded = metrics.get("models_succeeded")
            transfer_succeeded = metrics.get("transfer_models_succeeded")
            all_inference_failed = (
                isinstance(requested, int) and requested > 0 and succeeded == 0
            ) or ("transfer_models_succeeded" in metrics and transfer_succeeded == 0)
            if all_inference_failed:
                status["status"] = "failed"
            else:
                partial_inference = isinstance(requested, int) and succeeded != requested
                status["status"] = "completed_with_errors" if errors or partial_inference else "completed"
            event_log("run_completed", status=status["status"], error_count=len(errors))
        except KeyboardInterrupt as exc:
            result = {
                "metrics": {},
                "raw_outputs": {},
                "artifacts": {},
                "errors": [{
                    "stage": "run",
                    "type": type(exc).__name__,
                    "message": "Run interrupted by KeyboardInterrupt.",
                    "traceback": traceback.format_exc(),
                }],
            }
            status["status"] = "failed"
            status["error"] = {
                "type": type(exc).__name__,
                "message": "Run interrupted by KeyboardInterrupt.",
                "traceback": traceback.format_exc(),
            }
            event_log("run_failed", error_type=type(exc).__name__, message="Run interrupted by KeyboardInterrupt.")
        except Exception as exc:
            formatted_traceback = traceback.format_exc()
            result = {
                "metrics": {},
                "raw_outputs": {},
                "artifacts": {},
                "errors": [{
                    "stage": "run",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": formatted_traceback,
                }],
            }
            status["status"] = "failed"
            status["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": formatted_traceback,
            }
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
    return status["status"] == "completed", run_dir


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
        ok, run_dir = execute_run(config, f"src/experiment_runner.py run {config_path}")
        if ok:
            return 0
        status = load_json(run_dir / "status.json")
        return 130 if status.get("error", {}).get("type") == "KeyboardInterrupt" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _batch(args: argparse.Namespace) -> int:
    batch_path = Path(args.batch).expanduser().resolve()
    raw = load_json(batch_path)
    entries = batch_entries(raw, batch_path)
    prepared: list[tuple[dict[str, Any], Path]] = []
    workflow_families: set[str] = set()
    validation_errors: list[str] = []
    for index, entry in enumerate(entries):
        config_path = Path(entry["config"])
        try:
            config = _load_resolved(
                config_path,
                [f"{key}={json.dumps(value)}" for key, value in entry.get("set", {}).items()],
            )
            prepared.append((config, config_path))
            workflow_families.add("ocr" if config["workflow"].startswith("ocr") else "vlm")
        except Exception as exc:
            validation_errors.append(f"Batch entry {index} ({config_path}) invalid: {type(exc).__name__}: {exc}")
    if validation_errors:
        for message in validation_errors:
            print(message, file=sys.stderr)
        print("No batch runs were started because validation failed.", file=sys.stderr)
        return 1
    if len(workflow_families) > 1:
        print(
            "Batch entries must use one workflow environment: do not mix OCR and VLM workflows in one batch.",
            file=sys.stderr,
        )
        return 1
    failures = 0
    completed: list[str] = []
    for index, (config, config_path) in enumerate(prepared):
        try:
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
        results_path = run_dir / "results.json"
        results = load_json(results_path) if results_path.is_file() else {}
        pid = status.get("pid")
        hostname = status.get("hostname")
        stale = False
        if status.get("status") == "running" and hostname == socket.gethostname() and isinstance(pid, int):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                stale = True
            except PermissionError:
                # The process may exist but be owned by another user.
                stale = False
        runs.append(
            {
                "run_id": status.get("run_id", run_dir.name),
                "workflow": status.get("workflow"),
                "status": status.get("status"),
                "metrics": results.get("metrics", {}),
                "scalar_metrics": _flatten_scalar_metrics(results.get("metrics", {})),
                "stale": stale,
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
            keys = set(run["scalar_metrics"])
            common = keys if common is None else common & keys
        metric_keys = sorted(common or [])
        compatible_groups.append(
            {
                "workflow": workflow,
                "metric_keys": metric_keys,
                "metric_values": {
                    run["run_id"]: {key: run["scalar_metrics"][key] for key in metric_keys}
                    for run in group
                },
                "run_ids": [run["run_id"] for run in group],
            }
        )
    payload = {"runs": runs, "compatible_groups": compatible_groups}
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    lines = ["# Experiment summary", ""]
    for run in runs:
        stale_note = " (stale local process)" if run["stale"] else ""
        lines.extend([
            f"## `{run['run_id']}` — `{run['workflow']}` — `{run['status']}`{stale_note}",
            "",
        ])
        if run["scalar_metrics"]:
            for key in sorted(run["scalar_metrics"]):
                lines.append(f"- `{key}`: {json.dumps(run['scalar_metrics'][key], ensure_ascii=False)}")
        else:
            lines.append("- No scalar metrics recorded.")
        lines.append("")
    lines.extend(["Runs are compared only within the same workflow using shared scalar metric paths; no universal score or external judge is used."])
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
