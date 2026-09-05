# Experiment Protocol

## Research question

So sánh YOLO26s và RT-DETR-L trên cùng một bài toán vehicle detection giao thông Việt Nam theo bốn trục:

1. detection accuracy;
2. per-class robustness;
3. computational efficiency;
4. qualitative failure behavior.

## Fixed controls

- Same dataset version and split.
- Same 640 input resolution.
- Same 50-epoch budget.
- Same batch size 4.
- Same random seed 42.
- Same common augmentations.
- Same test split for final metrics.
- Same T4 and same preloaded images for speed benchmark.

## Primary metric

`mAP50-95` on the held-out test split.

Secondary metrics:

- P / R / F1;
- mAP50 / mAP75;
- per-class P / R / F1 / AP50 / AP75 / mAP50-95;
- latency mean/median/p95;
- FPS;
- checkpoint MB;
- parameter count;
- CUDA peak allocated/reserved memory.

## Visual evidence

Dataset:

- split distribution;
- class distribution;
- labeled sample grid.

Training:

- standard Ultralytics `results.png`;
- train batches;
- raw `results.csv`.

Test evaluation:

- PR curve;
- P curve;
- R curve;
- F1 curve;
- confusion matrix;
- normalized confusion matrix;
- validation/test batch labels and predictions.

Qualitative:

- deterministic side-by-side samples: GT | YOLO26 | RT-DETR;
- lowest diagnostic-F1 examples for each model.

## Important metric separation

AP evaluation uses a low confidence threshold (`0.001`) so the full PR ranking is available.

Failure diagnostics use a fixed deployment-style threshold (`conf=0.25`, IoU match `0.50`). These per-image TP/FP/FN/P/R/F1 values are only for error inspection and must not be presented as replacements for AP.

## Interpretation

Because YOLO26s and RT-DETR-L are not parameter-matched, a conclusion such as “model A is universally better” is not supported by this experiment. The appropriate conclusion is a trade-off statement based on accuracy, speed, memory, capacity and failure cases.
