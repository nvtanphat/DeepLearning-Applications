from __future__ import annotations

import argparse

from src.audit_dataset import audit_dataset
from src.benchmark import benchmark_model
from src.download_data import download_dataset
from src.evaluate import evaluate_model
from src.report import generate_report
from src.train import train_model
from src.visualize_predictions import generate_prediction_visuals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-qualitative", action="store_true")
    args = p.parse_args()

    if not args.skip_download:
        print("[1/7] Download dataset"); download_dataset(args.config)
    print("[2/7] Audit + dataset visualizations"); audit_dataset(args.config)

    from src.common import load_config
    cfg = load_config(args.config)
    model_keys = list(cfg["models"].keys())

    if not args.skip_train:
        for idx, key in enumerate(model_keys, 1):
            print(f"[Train {idx}/{len(model_keys)}] Train {key.upper()}")
            train_model(args.config, key)

    print("[Evaluation & Benchmark] Full test evaluation + benchmark")
    for key in model_keys:
        evaluate_model(args.config, key)
        benchmark_model(args.config, key)

    if not args.skip_qualitative:
        print("[6/7] Side-by-side predictions + worst-case diagnostics")
        generate_prediction_visuals(args.config)

    print("[7/7] Generate final report")
    print(generate_report(args.config))


if __name__ == "__main__":
    main()
