from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common import create_model, ensure_dirs, find_data_yaml, load_config, resolve_path


def training_args(cfg: dict[str, Any], model_key: str, data_yaml: Path) -> dict[str, Any]:
    exp = cfg["experiment"]
    aug = cfg.get("augmentation", {})
    mcfg = cfg["models"][model_key]
    args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(exp["epochs"]),
        "imgsz": int(exp["imgsz"]),
        "batch": int(exp["batch"]),
        "seed": int(exp["seed"]),
        "workers": int(exp["workers"]),
        "patience": int(exp["patience"]),
        "device": exp["device"],
        "cache": bool(exp["cache"]),
        "optimizer": exp.get("optimizer", "auto"),
        "amp": bool(mcfg.get("amp", True)),
        "deterministic": bool(mcfg.get("deterministic", True)),
        "project": str(resolve_path(cfg["paths"]["runs_dir"])),
        "name": mcfg["run_name"],
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }
    for key in ("lr0", "weight_decay"):
        if exp.get(key) is not None:
            args[key] = exp[key]
    args.update(aug)
    return args


def train_model(config_path: str, model_key: str, resume: str | None = None) -> Path:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    data_yaml = find_data_yaml(cfg["dataset"]["output_dir"])
    mcfg = cfg["models"][model_key]

    if resume:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = resolve_path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        model = create_model(mcfg["family"], str(resume_path))
        model.train(
            resume=True,
            data=str(data_yaml),
            device=cfg["experiment"]["device"],
            batch=int(cfg["experiment"]["batch"]),
            workers=int(cfg["experiment"]["workers"]),
            cache=bool(cfg["experiment"]["cache"]),
            plots=True,
        )
    else:
        model = create_model(mcfg["family"], mcfg["checkpoint"])
        model.train(**training_args(cfg, model_key, data_yaml))

    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found after training: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", choices=["yolo26", "rtdetr", "all"], default="all")
    p.add_argument("--resume", help="Path to last.pt; only valid with a single --model")
    args = p.parse_args()

    keys = ["yolo26", "rtdetr"] if args.model == "all" else [args.model]
    if args.resume and len(keys) != 1:
        raise SystemExit("--resume requires a single model")
    for key in keys:
        ckpt = train_model(args.config, key, args.resume)
        print(f"{key}: {ckpt}")


if __name__ == "__main__":
    main()
