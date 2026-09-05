from __future__ import annotations

import argparse
from pathlib import Path

from src.common import create_model, load_config, resolve_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Image, video, folder, webcam index, or URL")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True, help="Model key (e.g. yolo26, yolo26n, yolo26m, rtdetr)")
    p.add_argument("--checkpoint")
    p.add_argument("--conf", type=float)
    args = p.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["models"][args.model]
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "weights" / "best.pt"
    if not ckpt.is_absolute():
        ckpt = resolve_path(ckpt)

    model = create_model(mcfg["family"], str(ckpt))
    exp = cfg["experiment"]
    source = int(args.source) if args.source.isdigit() else args.source
    model.predict(
        source=source,
        imgsz=int(exp["imgsz"]),
        conf=float(args.conf if args.conf is not None else exp["conf"]),
        device=exp["device"],
        save=True,
        project=str(resolve_path(cfg["paths"]["predictions_dir"])),
        name=args.model,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
