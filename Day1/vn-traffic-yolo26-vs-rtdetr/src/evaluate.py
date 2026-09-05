from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.common import create_model, find_data_yaml, json_dump, load_config, resolve_path, safe_float


def _names_dict(metrics: Any) -> dict[int, str]:
    names = getattr(metrics, "names", {}) or {}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _per_class_rows(metrics: Any) -> list[dict[str, Any]]:
    box = getattr(metrics, "box", None)
    if box is None:
        return []
    names = _names_dict(metrics)
    raw_indices = getattr(box, "ap_class_index", None)
    indices = list(raw_indices) if raw_indices is not None else []
    p = np.asarray(getattr(box, "p", []), dtype=float)
    r = np.asarray(getattr(box, "r", []), dtype=float)
    f1 = np.asarray(getattr(box, "f1", []), dtype=float)
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    ap = np.asarray(getattr(box, "ap", []), dtype=float)
    all_ap = np.asarray(getattr(box, "all_ap", []), dtype=float)
    rows: list[dict[str, Any]] = []
    for j, cls_idx in enumerate(indices):
        cls = int(cls_idx)
        ap75 = float(all_ap[j, 5]) if all_ap.ndim == 2 and all_ap.shape[1] > 5 and j < len(all_ap) else None
        rows.append({
            "class_id": cls,
            "class_name": names.get(cls, str(cls)),
            "precision": float(p[j]) if j < len(p) else None,
            "recall": float(r[j]) if j < len(r) else None,
            "f1": float(f1[j]) if j < len(f1) else None,
            "AP50": float(ap50[j]) if j < len(ap50) else None,
            "AP75": ap75,
            "mAP50-95": float(ap[j]) if j < len(ap) else None,
        })
    return rows


def _metrics_to_dict(metrics: Any, model_key: str, checkpoint: Path, params: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model_key,
        "checkpoint": str(checkpoint),
        "params": params,
        "speed_ms": dict(getattr(metrics, "speed", {}) or {}),
    }
    result_dict = getattr(metrics, "results_dict", {}) or {}
    for k, v in result_dict.items():
        fv = safe_float(v)
        out[k] = fv if fv is not None else str(v)
    box = getattr(metrics, "box", None)
    if box is not None:
        out["precision"] = safe_float(getattr(box, "mp", None))
        out["recall"] = safe_float(getattr(box, "mr", None))
        p, r = out["precision"], out["recall"]
        out["f1"] = (2 * p * r / (p + r)) if p is not None and r is not None and (p + r) > 0 else None
        out["mAP50-95"] = safe_float(getattr(box, "map", None))
        out["mAP50"] = safe_float(getattr(box, "map50", None))
        out["mAP75"] = safe_float(getattr(box, "map75", None))
        maps = getattr(box, "maps", None)
        if maps is not None:
            names = _names_dict(metrics)
            out["per_class_mAP50-95"] = {names.get(i, str(i)): float(value) for i, value in enumerate(maps)}
    return out


def evaluate_model(config_path: str, model_key: str, checkpoint: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    mcfg = cfg["models"][model_key]
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    ckpt = Path(checkpoint) if checkpoint else run_dir / "weights" / "best.pt"
    if not ckpt.is_absolute():
        ckpt = resolve_path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model = create_model(mcfg["family"], str(ckpt))
    params = sum(p.numel() for p in model.model.parameters())
    data_yaml = find_data_yaml(cfg["dataset"]["output_dir"])
    exp = cfg["experiment"]
    val_dir = resolve_path(cfg["paths"]["reports_dir"]) / "val_runs"
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(exp["imgsz"]),
        batch=int(exp["batch"]),
        device=exp["device"],
        conf=float(exp.get("val_conf", 0.001)),
        iou=float(exp["iou"]),
        plots=True,
        project=str(val_dir),
        name=model_key,
        exist_ok=True,
        verbose=True,
    )
    report = _metrics_to_dict(metrics, model_key, ckpt, params)
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    json_dump(report, reports_dir / f"{model_key}_metrics.json")

    rows = _per_class_rows(metrics)
    per_class_path = reports_dir / f"{model_key}_per_class_metrics.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["class_id", "class_name", "precision", "recall", "f1", "AP50", "AP75", "mAP50-95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report["per_class_metrics_csv"] = str(per_class_path)
    report["validation_artifacts_dir"] = str(val_dir / model_key)
    json_dump(report, reports_dir / f"{model_key}_metrics.json")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", default="all", help="Model key or 'all'")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    cfg = load_config(args.config)
    available = list(cfg["models"].keys())
    if args.model == "all":
        keys = available
    elif args.model in cfg["models"]:
        keys = [args.model]
    else:
        raise SystemExit(f"Unknown model: {args.model}. Available: {available}")
    if args.checkpoint and len(keys) != 1:
        raise SystemExit("--checkpoint requires a single model")
    for key in keys:
        print(evaluate_model(args.config, key, args.checkpoint))


if __name__ == "__main__":
    main()
