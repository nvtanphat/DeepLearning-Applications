# Kaggle CLI Audit

## 1. Headless dataset access

The Kaggle kernel does not depend on `ROBOFLOW_API_KEY` or `UserSecretsClient`.

Workflow:

1. Download Roboflow dataset locally.
2. Package it as a private Kaggle Dataset.
3. Attach that dataset through `dataset_sources`.
4. Kernel reads from `/kaggle/input`.

This avoids a hidden manual UI dependency.

## 2. Kernel metadata

`kernel-metadata.template.json` uses native JSON booleans:

```json
"is_private": true,
"enable_gpu": true,
"enable_internet": true
```

The push command explicitly requests:

```text
--accelerator NvidiaTeslaT4
```

The generated metadata is tested before packaging.

## 3. Dataset integrity gate

Before training, the kernel verifies the fixed dataset version contract:

- total images: 3376;
- train: 2062;
- val: 574;
- test: 740;
- classes: 8;
- invalid YOLO label lines: 0.

If these checks fail, training stops instead of silently benchmarking on a different export.

## 4. Output persistence

Raw dataset extraction uses `/tmp/vn_traffic_data`; this prevents thousands of source images from being copied into kernel outputs.

Persistent experiment outputs stay under `/kaggle/working`:

```text
runs/
reports/
```

These can be downloaded with `kaggle kernels output`.

## 5. Resume across Kaggle runs

A new Kaggle run does not rely on the previous `/kaggle/working` directory.

Instead:

1. download previous kernel output;
2. package `last.pt` / `best.pt` into a private Kaggle Dataset;
3. attach the resume dataset on the next push;
4. inspect checkpoint epoch count;
5. resume unfinished models and skip already-completed models.

This handles the case where YOLO26 finishes but RT-DETR is interrupted.

## 6. Evaluation/report contract

Each completed Kaggle run produces:

- full test P/R/F1/mAP metrics;
- per-class P/R/F1/AP50/AP75/mAP50-95;
- standard Ultralytics PR/F1/P/R curves;
- raw + normalized confusion matrix;
- training curves and train-batch images;
- deterministic GT/YOLO26/RT-DETR side-by-side samples;
- lowest diagnostic-F1 failure cases;
- latency/FPS/checkpoint size/peak CUDA memory;
- final `reports/REPORT.md`.

## 7. What cannot be validated offline

The repository can statically validate code, metadata generation, packaging logic, and report logic. It cannot prove the user's Kaggle GPU quota, credentials, or account-specific execution availability without running on that account.
