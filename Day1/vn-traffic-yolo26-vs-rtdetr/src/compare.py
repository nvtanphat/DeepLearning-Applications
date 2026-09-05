from __future__ import annotations

import argparse
from pathlib import Path

from src.report import generate_report


def compare(config_path: str) -> tuple[Path, Path]:
    report = generate_report(config_path)
    return report.parent / "comparison.csv", report


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/experiment.yaml"); args = p.parse_args()
    print(compare(args.config))


if __name__ == "__main__":
    main()
