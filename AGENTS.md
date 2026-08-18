# Codex Agent Instructions

## Core Philosophy
- Prefer **simple, explicit solutions** over abstractions.
- Do **not** introduce new design patterns unless explicitly requested.
- Do **not** refactor unrelated code.
- Do **not** add new files unless necessary.
- Do **not** generalize for hypothetical future use cases.

If a solution works for the current requirements, it is sufficient.

## Scope of Changes
- Modify **only** the files explicitly mentioned in the request.
- Keep diffs minimal and localized.
- Avoid renaming variables, functions, or files unless required for correctness.

## Architecture Rules
- Respect the existing architecture.
- Do not introduce:
  - new layers
  - factories
  - dependency injection
  - registries
  - abstractions
  - frameworks
unless explicitly requested.

## Coding Style
- Match the existing code style exactly.
- Prefer:
  - direct logic over helper indirection
- Avoid cleverness.

## Performance & Correctness
- Optimize only when:
  - there is a demonstrated bottleneck, or
  - optimization is explicitly requested.
- Prioritize correctness and readability over performance tweaks, unless performance is explicitly requested.

## Tests
- Do not add tests unless explicitly requested.
- Do not modify existing tests unless they are failing due to the requested change.

## Documentation & Comments
- Write moderate comments, keep them as succint as possible

## Output Expectations
- Provide the **simplest working solution**.
- Explain changes briefly if asked.
- If multiple solutions exist, choose the most conservative one.

## Minimal Changes
- Make the minimal set of code changes necessary to correctly implement the requested logic.
- Do not modify unrelated code, refactor unnecessarily, or add speculative improvements.
- Preserve existing behavior outside the requested scope.

## When Unsure
- Ask a clarification question instead of guessing.

## Experiment workflow environment contract

Use the canonical CLI in `src/experiment_runner.py` for experiments. OCR workflows must run with the OCR Conda environment:

```bash
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py run CONFIG.json
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py batch BATCH.json
```

VLM workflows must run with the `llm-misuse` Conda environment:

```bash
/home/jmadden2/anaconda3/envs/llm-misuse/bin/python src/experiment_runner.py run CONFIG.json
/home/jmadden2/anaconda3/envs/llm-misuse/bin/python src/experiment_runner.py batch BATCH.json --fail-fast
```

Do not use bare `python` or the base Conda environment for experiment execution. Install dependencies and troubleshoot only in the environment belonging to the selected workflow; do not modify the other workflow environment. Environment-neutral commands such as `list-models`, `validate`, and deterministic `summarize` do not import model libraries and may use either workflow interpreter.
