from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import cv2

from src.common import create_model, find_data_yaml, image_files, json_dump, load_config, resolve_path, split_images_dir


def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_model(config_path: str, model_key: str, checkpoint: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    mcfg = cfg["models"][model_key]
    exp = cfg["experiment"]
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    ckpt = Path(checkpoint) if checkpoint else run_dir / "weights" / "best.pt"
    if not ckpt.is_absolute():
        ckpt = resolve_path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = create_model(mcfg["family"], str(ckpt))

    data_yaml = find_data_yaml(cfg["dataset"]["output_dir"])
    test_dir = split_images_dir(data_yaml, "test")
    images = image_files(test_dir)
    if not images:
        raise RuntimeError(f"No test images in {test_dir}")

    warmup_n = min(int(exp["warmup_images"]), len(images))
    test_n = min(int(exp["benchmark_images"]), len(images))
    frames = [cv2.imread(str(p)) for p in images[:test_n]]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("Could not decode benchmark images")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # type: ignore

    for frame in frames[:warmup_n]:
        model.predict(frame, imgsz=int(exp["imgsz"]), device=exp["device"], verbose=False)
    _sync_cuda()

    timings = []
    for frame in frames:
        _sync_cuda()
        t0 = time.perf_counter()
        model.predict(frame, imgsz=int(exp["imgsz"]), device=exp["device"], verbose=False)
        _sync_cuda()
        timings.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(timings)
    peak_alloc_mb = None
    peak_reserved_mb = None
    try:
        if torch is not None and torch.cuda.is_available():
            peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
            peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    except Exception:
        pass

    report = {
        "model": model_key,
        "checkpoint": str(ckpt),
        "checkpoint_size_MB": ckpt.stat().st_size / 1024**2,
        "images": len(timings),
        "warmup_images": warmup_n,
        "mean_ms_per_image": mean_ms,
        "median_ms_per_image": statistics.median(timings),
        "p95_ms_per_image": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "min_ms_per_image": min(timings),
        "max_ms_per_image": max(timings),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else None,
        "peak_cuda_allocated_MB": peak_alloc_mb,
        "peak_cuda_reserved_MB": peak_reserved_mb,
        "batch_size": 1,
        "imgsz": int(exp["imgsz"]),
        "disk_io_included": False,
    }
    json_dump(report, resolve_path(cfg["paths"]["reports_dir"]) / f"{model_key}_benchmark.json")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", choices=["yolo26", "rtdetr", "all"], default="all")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    keys = ["yolo26", "rtdetr"] if args.model == "all" else [args.model]
    if args.checkpoint and len(keys) != 1:
        raise SystemExit("--checkpoint requires a single model")
    for key in keys:
        print(benchmark_model(args.config, key, args.checkpoint))


if __name__ == "__main__":
    main()
