from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {cfg_path}")
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("runs_dir", "reports_dir", "predictions_dir", "exports_dir"):
        resolve_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg["dataset"]["output_dir"]).mkdir(parents=True, exist_ok=True)


def find_data_yaml(dataset_dir: str | Path) -> Path:
    root = resolve_path(dataset_dir)
    candidates = [root / "data.yaml", root / "dataset.yaml"]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(root.rglob("data.yaml")) + sorted(root.rglob("dataset.yaml"))
    if not matches:
        raise FileNotFoundError(f"No data.yaml found under {root}")
    return matches[0]


def roboflow_api_key() -> str:
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass
    raise RuntimeError(
        "Missing ROBOFLOW_API_KEY. Set it as an environment variable or a Kaggle secret."
    )


def create_model(family: str, checkpoint: str):
    family = family.lower()
    if family == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(checkpoint)
    if family == "yolo":
        from ultralytics import YOLO

        return YOLO(checkpoint)
    raise ValueError(f"Unsupported model family: {family}")


def json_dump(obj: Any, path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_images_dir(data_yaml: Path, split: str) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if split not in data_cfg:
        raise KeyError(f"Split '{split}' not present in {data_yaml}")

    base = data_yaml.parent
    if data_cfg.get("path"):
        root = Path(data_cfg["path"])
        if not root.is_absolute():
            root = (base / root).resolve()
    else:
        root = base

    value = data_cfg[split]
    if isinstance(value, list):
        value = value[0]
    p = Path(value)
    if not p.is_absolute():
        candidate = (root / p).resolve()
        if not candidate.exists():
            clean = str(value).lstrip("./\\").replace("\\", "/")
            fallback = (base / clean).resolve()
            if fallback.exists():
                return fallback
        p = candidate
    return p


def image_files(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
