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
    for key in cfg["models"]:
        metrics_file = reports / f"{key}_metrics.json"
        bench_file = reports / f"{key}_benchmark.json"
        if not metrics_file.exists() or not bench_file.exists():
            continue
        metrics = _load_json(metrics_file)
        bench = _load_json(bench_file)
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
    per_class_dfs = []
    for key in cfg["models"]:
        p_csv = reports / f"{key}_per_class_metrics.csv"
        if p_csv.exists():
            df_k = pd.read_csv(p_csv).add_prefix(f"{key}_")
            per_class_dfs.append(df_k)
    if per_class_dfs:
        merged = per_class_dfs[0]
        first_col = [c for c in merged.columns if c.endswith("_class_id")][0]
        for df_other in per_class_dfs[1:]:
            other_col = [c for c in df_other.columns if c.endswith("_class_id")][0]
            merged = merged.merge(df_other, left_on=first_col, right_on=other_col, how="outer")
        merged.to_csv(reports / "per_class_comparison.csv", index=False)

    lines = [
        "# Model Comparison — Vietnam Vehicle Detection",
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
        "![Dataset splits](dataset/split_distribution.png)",
        "![Class distribution](dataset/class_distribution.png)",
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
        "See `per_class_comparison.csv` for precision, recall, F1, AP50, AP75, and mAP50-95 by class.",
        "",
        "## 5. Standard validation visuals",
        "",
        "Ultralytics-generated validation artifacts are retained per model: PR/F1/P/R curves, confusion matrices (raw and normalized), validation labels and predictions.",
        "",
    ]
    for key in cfg["models"]:
        lines.extend([
            f"### {key.upper()}",
            "",
            f"![{key} PR](val_runs/{key}/BoxPR_curve.png)",
            f"![{key} F1](val_runs/{key}/BoxF1_curve.png)",
            f"![{key} confusion](val_runs/{key}/confusion_matrix_normalized.png)",
            "",
        ])
    lines.extend([
        "## 6. Training curves",
        "",
        "Training images and `results.csv` are copied into `models/<model>/training/`. The standard `results.png` captures loss and validation metric trajectories.",
        "",
    ])
    for key in cfg["models"]:
        lines.append(f"![{key} training](models/{key}/training/results.png)")
    lines.extend([
        "",
        "## 7. Qualitative comparison and error analysis",
        "",
        "Deterministic side-by-side samples are under `qualitative/side_by_side/`.",
        "Worst cases are under `qualitative/worst_<model>/`. Their TP/FP/FN/P/R/F1 use a fixed diagnostic confidence/IoU threshold and are for error analysis, not AP reporting.",
        "",
        "## 8. Deployment metrics",
        "",
        "Benchmark JSON files include mean/median/p95 latency, FPS, checkpoint size, and peak CUDA allocated/reserved memory where CUDA is available.",
        "",
        "## 9. Interpretation rule",
        "",
        "Do not declare a winner from mAP alone. Report accuracy, per-class behavior, failure cases, latency/FPS, VRAM, checkpoint size, and parameter count together. The chosen variants are not parameter-matched.",
    ])
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
