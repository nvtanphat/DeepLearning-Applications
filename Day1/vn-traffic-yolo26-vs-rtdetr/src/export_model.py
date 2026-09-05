from __future__ import annotations

import argparse
from pathlib import Path

from src.common import create_model, load_config, resolve_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True, help="Model key (e.g. yolo26, yolo26n, yolo26m, rtdetr)")
    p.add_argument("--checkpoint")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino"])
    args = p.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["models"][args.model]
    run_dir = resolve_path(cfg["paths"]["runs_dir"]) / mcfg["run_name"]
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "weights" / "best.pt"
    if not ckpt.is_absolute():
        ckpt = resolve_path(ckpt)
    model = create_model(mcfg["family"], str(ckpt))
    out = model.export(format=args.format, imgsz=int(cfg["experiment"]["imgsz"]), device=cfg["experiment"]["device"])
    print(out)


if __name__ == "__main__":
    main()
