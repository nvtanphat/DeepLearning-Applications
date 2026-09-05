from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.audit_dataset import label_dir_from_images, label_path_for_image
from src.common import create_model, find_data_yaml, image_files, load_config, resolve_path, split_images_dir


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: Path, w_img: int, h_img: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0])); x, y, w, h = map(float, parts[1:])
            boxes.append([(x - w / 2) * w_img, (y - h / 2) * h_img, (x + w / 2) * w_img, (y + h / 2) * h_img])
            classes.append(cls)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(classes, dtype=int)


def _match(gt_boxes: np.ndarray, gt_cls: np.ndarray, pred_boxes: np.ndarray, pred_cls: np.ndarray, iou_thr: float = 0.5) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for gi in range(len(gt_boxes)):
        for pi in range(len(pred_boxes)):
            if int(gt_cls[gi]) != int(pred_cls[pi]):
                continue
            i = _iou(gt_boxes[gi], pred_boxes[pi])
            if i >= iou_thr:
                candidates.append((i, gi, pi))
    candidates.sort(reverse=True)
    used_g: set[int] = set(); used_p: set[int] = set(); tp = 0
    for _, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def _title_tile(img: np.ndarray, title: str) -> np.ndarray:
    tile = cv2.resize(img, (420, 315), interpolation=cv2.INTER_AREA)
    band = np.zeros((42, tile.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, title[:60], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([band, tile])


def _draw_gt(img: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, names: dict[int, str]) -> np.ndarray:
    out = img.copy()
    for box, cls in zip(gt_boxes, gt_cls):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, names.get(int(cls), str(int(cls))), (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def generate_prediction_visuals(config_path: str, sample_count: int | None = None, worst_count: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    viz_cfg = cfg.get("visualization", {})
    sample_count = int(sample_count or viz_cfg.get("sample_count", 12))
    worst_count = int(worst_count or viz_cfg.get("worst_count", 6))
    conf = float(viz_cfg.get("diagnostic_conf", exp.get("conf", 0.25)))
    match_iou = float(viz_cfg.get("diagnostic_iou", 0.5))

    data_yaml = find_data_yaml(cfg["dataset"]["output_dir"])
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names_raw = data_cfg.get("names", {})
    if isinstance(names_raw, list):
        names = {i: str(n) for i, n in enumerate(names_raw)}
    else:
        names = {int(k): str(v) for k, v in names_raw.items()}

    test_dir = split_images_dir(data_yaml, "test")
    labels_dir = label_dir_from_images(test_dir)
    images = image_files(test_dir)
    if not images:
        raise RuntimeError(f"No test images in {test_dir}")

    # Deterministic evenly spaced sample, not hand-picked.
    if len(images) <= sample_count:
        samples = images
    else:
        idx = np.linspace(0, len(images) - 1, sample_count, dtype=int)
        samples = [images[i] for i in idx]

    models = {}
    for key in cfg["models"]:
        mcfg = cfg["models"][key]
        ckpt = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"] / "weights" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        models[key] = create_model(mcfg["family"], str(ckpt))

    out_root = resolve_path(cfg["paths"]["reports_dir"]) / "qualitative"
    side_dir = out_root / "side_by_side"
    worst_dirs = {k: out_root / f"worst_{k}" for k in models}
    side_dir.mkdir(parents=True, exist_ok=True)
    for d in worst_dirs.values(): d.mkdir(parents=True, exist_ok=True)

    diagnostic_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    sample_outputs: list[str] = []

    def predict_one(key: str, path: Path):
        result = models[key].predict(str(path), imgsz=int(exp["imgsz"]), conf=conf, iou=float(exp["iou"]), device=exp["device"], verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None and len(result.boxes) else np.empty((0, 4))
        cls = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None and len(result.boxes) else np.empty((0,), dtype=int)
        return result, boxes, cls

    # Diagnostics on all test images so worst-case selection is honest.
    cache: dict[tuple[str, str], Any] = {}
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None: continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = _load_gt(label_path_for_image(image_path, test_dir, labels_dir), w, h)
        for key in models:
            result, pred_boxes, pred_cls = predict_one(key, image_path)
            tp, fp, fn = _match(gt_boxes, gt_cls, pred_boxes, pred_cls, match_iou)
            p = tp / (tp + fp) if tp + fp else (1.0 if len(gt_boxes) == 0 else 0.0)
            r = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            diagnostic_rows[key].append({"image": image_path.name, "path": str(image_path), "gt": len(gt_boxes), "pred": len(pred_boxes), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1})
            if image_path in samples:
                cache[(key, str(image_path))] = result

    # Side-by-side deterministic samples.
    for i, image_path in enumerate(samples, 1):
        img = cv2.imread(str(image_path))
        if img is None: continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = _load_gt(label_path_for_image(image_path, test_dir, labels_dir), w, h)
        gt_img = _draw_gt(img, gt_boxes, gt_cls, names)
        tiles = [_title_tile(gt_img, "Ground truth")]
        for k in models:
            res = cache.get((k, str(image_path)))
            if res is None:
                res = predict_one(k, image_path)[0]
            tiles.append(_title_tile(res.plot(), k.upper()))
        canvas = np.hstack(tiles)
        out = side_dir / f"sample_{i:02d}_{image_path.stem[:40]}.jpg"
        cv2.imwrite(str(out), canvas)
        sample_outputs.append(str(out))

    # CSV + worst-case visualization per model.
    for key, rows in diagnostic_rows.items():
        csv_path = out_root / f"{key}_image_diagnostics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
            writer.writeheader(); writer.writerows(rows)
        worst = sorted(rows, key=lambda x: (x["f1"], -x["fn"], -x["fp"]))[:worst_count]
        for rank, row in enumerate(worst, 1):
            image_path = Path(row["path"])
            img = cv2.imread(str(image_path))
            if img is None: continue
            h, w = img.shape[:2]
            gt_boxes, gt_cls = _load_gt(label_path_for_image(image_path, test_dir, labels_dir), w, h)
            gt_img = _draw_gt(img, gt_boxes, gt_cls, names)
            result, _, _ = predict_one(key, image_path)
            title = f"{key} | F1={row['f1']:.2f} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
            canvas = np.hstack([_title_tile(gt_img, "Ground truth"), _title_tile(result.plot(), title)])
            cv2.imwrite(str(worst_dirs[key] / f"worst_{rank:02d}_{image_path.stem[:45]}.jpg"), canvas)

    return {"sample_outputs": sample_outputs, "diagnostic_conf": conf, "diagnostic_iou": match_iou, "output_dir": str(out_root)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--samples", type=int)
    p.add_argument("--worst", type=int)
    args = p.parse_args()
    print(generate_prediction_visuals(args.config, args.samples, args.worst))


if __name__ == "__main__":
    main()
