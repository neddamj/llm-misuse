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

Batches are homogeneous by environment: do not mix OCR and VLM workflows in one batch. Validate and run OCR batches with:

```bash
/home/jmadden2/anaconda3/envs/ocr/bin/python src/experiment_runner.py batch BATCH.json
```

Validate and run VLM batches with:

```bash
/home/jmadden2/anaconda3/envs/llm-misuse/bin/python src/experiment_runner.py batch BATCH.json --fail-fast
```

For VLM pipelines, `models.attack` selects attack workers and `models.transfer` selects sequential transfer models; transfer keys come from the VLM inference catalog and use `transfer_device`. `vlm_inference` uses only `device`. OCR inference keeps model-specific OCR prompts rather than applying the attack model's prompt to every model. `qianfan_ocr` and `hunyuan_ocr` are unavailable in the canonical OCR environment because their required Transformers classes are not installed.
