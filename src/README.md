# Agent workflow

The canonical entry point is `experiment_runner.py`. Generate an editable manifest, validate it, run it, inspect the saved JSON, and summarize compatible runs:

```bash
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py example --workflow ocr_pipeline --output /tmp/ocr.json
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py validate /tmp/ocr.json
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py run /tmp/ocr.json
less results/runs/<run-id>/status.json
less results/runs/<run-id>/results.json
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py summarize results/runs/<run-id> --format markdown
```

Use `/home/jmadden2/anaconda3/envs/llm-misuse/bin/python` for `vlm_attack`, `vlm_inference`, and `vlm_pipeline`. `run` saves the resolved configuration, immutable status/events, terminal log, normalized results, deterministic summary, and any image artifacts under `results/runs/`.
