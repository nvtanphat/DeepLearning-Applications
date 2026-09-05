from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, resolve_path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_training_artifacts(cfg: dict, key: str, reports: Path) -> Path:
    mcfg = cfg["models"][key]
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    dst = reports / "models" / key / "training"
    dst.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        for p in run_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv", ".yaml"}:
                shutil.copy2(p, dst / p.name)
    return dst


def _overall_plots(df: pd.DataFrame, reports: Path) -> None:
    figures = reports / "figures"; figures.mkdir(parents=True, exist_ok=True)
    metrics = ["precision", "recall", "f1", "mAP50", "mAP75", "mAP50-95"]
    plot_df = df.set_index("model")[metrics].T
    ax = plot_df.plot(kind="bar", figsize=(10, 5.5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Overall test metrics")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25)
    plt.tight_layout(); plt.savefig(figures / "overall_metrics.png", dpi=180); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, row in df.iterrows():
        ax.scatter(row["fps"], row["mAP50-95"], s=100)
        ax.annotate(row["model"], (row["fps"], row["mAP50-95"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("FPS (batch=1)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Accuracy-speed trade-off")
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(figures / "accuracy_speed_tradeoff.png", dpi=180); plt.close()

    if "peak_cuda_allocated_MB" in df.columns and df["peak_cuda_allocated_MB"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(df["model"], df["peak_cuda_allocated_MB"])
        ax.set_ylabel("Peak allocated CUDA memory (MB)")
        ax.set_title("Inference GPU memory")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig(figures / "gpu_memory.png", dpi=180); plt.close()


def generate_report(config_path: str) -> Path:
    cfg = load_config(config_path)
    reports = resolve_path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in ("yolo26", "rtdetr"):
        metrics = _load_json(reports / f"{key}_metrics.json")
        bench = _load_json(reports / f"{key}_benchmark.json")
        rows.append({
            "model": key,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "mAP50": metrics.get("mAP50"),
            "mAP75": metrics.get("mAP75"),
            "mAP50-95": metrics.get("mAP50-95"),
            "params_M": (metrics.get("params") or 0) / 1e6,
            "checkpoint_MB": bench.get("checkpoint_size_MB"),
            "mean_ms": bench.get("mean_ms_per_image"),
            "median_ms": bench.get("median_ms_per_image"),
            "p95_ms": bench.get("p95_ms_per_image"),
            "fps": bench.get("fps"),
            "peak_cuda_allocated_MB": bench.get("peak_cuda_allocated_MB"),
        })
        _copy_training_artifacts(cfg, key, reports)

    df = pd.DataFrame(rows)
    df.to_csv(reports / "comparison.csv", index=False)
    _overall_plots(df, reports)

    # Join per-class results for direct comparison.
    yc = pd.read_csv(reports / "yolo26_per_class_metrics.csv").add_prefix("yolo26_")
    rc = pd.read_csv(reports / "rtdetr_per_class_metrics.csv").add_prefix("rtdetr_")
    per_class = yc.merge(rc, left_on="yolo26_class_id", right_on="rtdetr_class_id", how="outer")
    per_class.to_csv(reports / "per_class_comparison.csv", index=False)

    lines = [
        "# YOLO26 vs RT-DETR — Vietnam Vehicle Detection",
        "",
        "## 1. Experimental protocol",
        "",
        f"- Dataset: {cfg['dataset']['name']} (fixed Roboflow version {cfg['dataset']['version']}).",
        f"- Image size: {cfg['experiment']['imgsz']} px; epochs: {cfg['experiment']['epochs']}; batch: {cfg['experiment']['batch']}; seed: {cfg['experiment']['seed']}.",
        "- Primary accuracy metric: mAP50-95 on the untouched test split.",
        "- AP evaluation uses low confidence threshold; qualitative diagnostics use the deployment confidence threshold and are not substituted for AP.",
        "- Latency benchmark is batch=1 on the same test images, after warmup, with disk I/O excluded.",
        "",
        "## 2. Dataset audit and visualization",
        "",
        "![Split distribution](dataset/split_distribution.png)",
        "",
        "![Class distribution](dataset/class_distribution.png)",
        "",
        "![Training label samples](dataset/train_label_samples.jpg)",
        "",
        "## 3. Overall quantitative results",
        "",
        df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![Overall metrics](figures/overall_metrics.png)",
        "",
        "![Accuracy-speed tradeoff](figures/accuracy_speed_tradeoff.png)",
        "",
        "## 4. Per-class metrics",
        "",
        "See `yolo26_per_class_metrics.csv`, `rtdetr_per_class_metrics.csv`, and `per_class_comparison.csv` for precision, recall, F1, AP50, AP75, and mAP50-95 by class.",
        "",
        "## 5. Standard validation visuals",
        "",
        "Ultralytics-generated validation artifacts are retained per model: PR/F1/P/R curves, confusion matrices (raw and normalized), validation labels and predictions.",
        "",
        "### YOLO26",
        "",
        "![YOLO26 PR](val_runs/yolo26/BoxPR_curve.png)",
        "![YOLO26 F1](val_runs/yolo26/BoxF1_curve.png)",
        "![YOLO26 confusion](val_runs/yolo26/confusion_matrix_normalized.png)",
        "",
        "### RT-DETR",
        "",
        "![RT-DETR PR](val_runs/rtdetr/BoxPR_curve.png)",
        "![RT-DETR F1](val_runs/rtdetr/BoxF1_curve.png)",
        "![RT-DETR confusion](val_runs/rtdetr/confusion_matrix_normalized.png)",
        "",
        "## 6. Training curves",
        "",
        "Training images and `results.csv` are copied into `models/<model>/training/`. The standard `results.png` captures loss and validation metric trajectories.",
        "",
        "![YOLO26 training](models/yolo26/training/results.png)",
        "![RT-DETR training](models/rtdetr/training/results.png)",
        "",
        "## 7. Qualitative comparison and error analysis",
        "",
        "Deterministic side-by-side samples are under `qualitative/side_by_side/`: Ground truth | YOLO26 | RT-DETR.",
        "Worst cases are under `qualitative/worst_yolo26/` and `qualitative/worst_rtdetr/`. Their TP/FP/FN/P/R/F1 use a fixed diagnostic confidence/IoU threshold and are for error analysis, not AP reporting.",
        "",
        "## 8. Deployment metrics",
        "",
        "Benchmark JSON files include mean/median/p95 latency, FPS, checkpoint size, and peak CUDA allocated/reserved memory where CUDA is available.",
        "",
        "## 9. Interpretation rule",
        "",
        "Do not declare a winner from mAP alone. Report accuracy, per-class behavior, failure cases, latency/FPS, VRAM, checkpoint size, and parameter count together. The two chosen variants are not parameter-matched.",
    ]
    if (reports / "figures" / "gpu_memory.png").exists():
        lines.insert(lines.index("## 9. Interpretation rule"), "![GPU memory](figures/gpu_memory.png)")
        lines.insert(lines.index("## 9. Interpretation rule"), "")

    out = reports / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (reports / "comparison.md").write_text("# Comparison\n\n" + df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/experiment.yaml"); args = p.parse_args()
    print(generate_report(args.config))


if __name__ == "__main__":
    main()
