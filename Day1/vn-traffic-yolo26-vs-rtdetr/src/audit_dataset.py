from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.common import find_data_yaml, image_files, json_dump, load_config, resolve_path, split_images_dir


def label_dir_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def audit_yolo_labels(labels_dir: Path, num_classes: int) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    label_files = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
    invalid_lines: list[dict[str, Any]] = []
    boxes = 0
    areas: list[float] = []
    aspect_ratios: list[float] = []

    for file in label_files:
        for line_no, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                invalid_lines.append({"file": str(file), "line": line_no, "reason": "expected 5 columns"})
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines.append({"file": str(file), "line": line_no, "reason": "non numeric"})
                continue
            if not (0 <= cls < num_classes):
                invalid_lines.append({"file": str(file), "line": line_no, "reason": f"class {cls} out of range"})
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid_lines.append({"file": str(file), "line": line_no, "reason": "bbox not normalized/positive"})
                continue
            class_counts[cls] += 1
            boxes += 1
            areas.append(w * h)
            aspect_ratios.append(w / h)

    return {
        "label_files": len(label_files),
        "boxes": boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "invalid_count": len(invalid_lines),
        "invalid_examples": invalid_lines[:50],
        "bbox_area": {
            "mean": float(np.mean(areas)) if areas else None,
            "median": float(np.median(areas)) if areas else None,
            "p10": float(np.percentile(areas, 10)) if areas else None,
            "p90": float(np.percentile(areas, 90)) if areas else None,
        },
        "aspect_ratio": {
            "mean": float(np.mean(aspect_ratios)) if aspect_ratios else None,
            "median": float(np.median(aspect_ratios)) if aspect_ratios else None,
        },
    }


def _read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls, x, y, w, h))
        except ValueError:
            continue
    return boxes


def _draw_gt(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], names: list[str]) -> np.ndarray:
    out = image.copy()
    h_img, w_img = out.shape[:2]
    for cls, x, y, w, h in boxes:
        x1 = max(0, int((x - w / 2) * w_img))
        y1 = max(0, int((y - h / 2) * h_img))
        x2 = min(w_img - 1, int((x + w / 2) * w_img))
        y2 = min(h_img - 1, int((y + h / 2) * h_img))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = names[cls] if 0 <= cls < len(names) else str(cls)
        cv2.putText(out, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def _make_dataset_visuals(report: dict[str, Any], names: list[str], data_yaml: Path, config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = resolve_path(cfg["paths"]["reports_dir"]) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split sizes.
    splits = list(report["splits"].keys())
    image_counts = [report["splits"][s]["images"] for s in splits]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(splits, image_counts)
    ax.set_title("Dataset split sizes")
    ax.set_ylabel("Images")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "split_distribution.png", dpi=180)
    plt.close(fig)

    # Class distribution across all splits.
    totals = Counter()
    for info in report["splits"].values():
        for cls, count in info["class_counts"].items():
            totals[int(cls)] += int(count)
    labels = [names[i] if i < len(names) else str(i) for i in range(len(names))]
    values = [totals.get(i, 0) for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title("Object instances per class")
    ax.set_ylabel("Bounding boxes")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    # Deterministic train-label sample grid.
    if "train" in report["splits"]:
        images_dir = Path(report["splits"]["train"]["images_dir"])
        labels_dir = Path(report["splits"]["train"]["labels_dir"])
        candidates = image_files(images_dir)
        rng = random.Random(int(cfg["experiment"].get("seed", 42)))
        rng.shuffle(candidates)
        chosen = candidates[:9]
        tiles: list[np.ndarray] = []
        for image_path in chosen:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            boxes = _read_boxes(label_path_for_image(image_path, images_dir, labels_dir))
            img = _draw_gt(img, boxes, names)
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(img, image_path.name[:34], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(img)
        if tiles:
            blank = np.zeros_like(tiles[0])
            while len(tiles) < 9:
                tiles.append(blank.copy())
            rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
            cv2.imwrite(str(out_dir / "train_label_samples.jpg"), np.vstack(rows))


def audit_dataset(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    data_yaml = find_data_yaml(cfg["dataset"]["output_dir"])
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data_cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    nc = len(names) if names else int(data_cfg.get("nc", 0))

    report: dict[str, Any] = {"data_yaml": str(data_yaml), "classes": names, "splits": {}}
    split_keys = {"train": "train", "val": "val" if "val" in data_cfg else "valid", "test": "test"}
    for split, yaml_split in split_keys.items():
        if yaml_split not in data_cfg:
            continue
        images_dir = split_images_dir(data_yaml, yaml_split)
        labels_dir = label_dir_from_images(images_dir)
        info = audit_yolo_labels(labels_dir, nc)
        images = image_files(images_dir) if images_dir.exists() else []
        info["images"] = len(images)
        info["missing_label_files"] = max(0, len(images) - info["label_files"])
        info["images_dir"] = str(images_dir)
        info["labels_dir"] = str(labels_dir)
        report["splits"][split] = info

    report["total_images"] = sum(s["images"] for s in report["splits"].values())
    report["total_boxes"] = sum(s["boxes"] for s in report["splits"].values())
    report["expected_images"] = cfg["dataset"].get("expected_images")
    report["expected_classes"] = cfg["dataset"].get("expected_classes")
    expected_splits = cfg["dataset"].get("expected_splits", {}) or {}
    split_matches = {
        split: report["splits"].get(split, {}).get("images") == expected
        for split, expected in expected_splits.items()
    }
    report["checks"] = {
        "image_count_matches_expected": report["total_images"] == report["expected_images"] if report["expected_images"] else None,
        "class_count_matches_expected": len(names) == report["expected_classes"] if report["expected_classes"] else None,
        "split_counts_match_expected": split_matches,
        "invalid_labels": sum(s["invalid_count"] for s in report["splits"].values()),
        "missing_label_files": sum(s["missing_label_files"] for s in report["splits"].values()),
    }
    failures = []
    if cfg["dataset"].get("strict_checks", False):
        if report["checks"]["image_count_matches_expected"] is False:
            failures.append(f"total images {report['total_images']} != expected {report['expected_images']}")
        if report["checks"]["class_count_matches_expected"] is False:
            failures.append(f"classes {len(names)} != expected {report['expected_classes']}")
        for split, ok in split_matches.items():
            if not ok:
                failures.append(f"{split} images {report['splits'].get(split, {}).get('images')} != expected {expected_splits[split]}")
        if report["checks"]["invalid_labels"]:
            failures.append(f"invalid label lines = {report['checks']['invalid_labels']}")
    report["checks"]["strict_failures"] = failures

    out_json = resolve_path(cfg["paths"]["reports_dir"]) / "dataset_audit.json"
    json_dump(report, out_json)

    csv_path = resolve_path(cfg["paths"]["reports_dir"]) / "class_distribution.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "class_name", "boxes"])
        for split, info in report["splits"].items():
            for cls, count in info["class_counts"].items():
                cls_i = int(cls)
                writer.writerow([split, cls_i, names[cls_i] if cls_i < len(names) else cls_i, count])

    _make_dataset_visuals(report, list(names), data_yaml, config_path)
    if failures:
        raise RuntimeError("Dataset integrity check failed: " + "; ".join(failures))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    args = p.parse_args()
    report = audit_dataset(args.config)
    print(f"Audit written. Classes={len(report['classes'])}, images={report['total_images']}, boxes={report['total_boxes']}")
    for split, info in report["splits"].items():
        print(f"{split}: images={info['images']} boxes={info['boxes']} invalid={info['invalid_count']}")


if __name__ == "__main__":
    main()
