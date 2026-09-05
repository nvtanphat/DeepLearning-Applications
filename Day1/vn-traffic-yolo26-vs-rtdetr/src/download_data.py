from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.common import ensure_dirs, find_data_yaml, json_dump, load_config, resolve_path, roboflow_api_key


def normalize_data_yaml(data_yaml: Path) -> None:
    with data_yaml.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        return

    base = data_yaml.parent
    changed = False
    for split in ["train", "val", "test"]:
        if split in content:
            val = content[split]
            if isinstance(val, str) and val.startswith(".."):
                clean = val.lstrip("./\\").replace("\\", "/")
                if (base / clean).exists():
                    content[split] = clean
                    changed = True
    if changed:
        with data_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)


def download_dataset(config_path: str, force: bool = False) -> Path:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    dcfg = cfg["dataset"]
    out_dir = resolve_path(dcfg["output_dir"])

    if not force:
        try:
            return find_data_yaml(out_dir)
        except FileNotFoundError:
            pass

    if force and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    rf = Roboflow(api_key=roboflow_api_key())
    project = rf.workspace(dcfg["workspace"]).project(dcfg["project"])
    version = project.version(int(dcfg["version"]))

    formats = [dcfg.get("export_format", "yolo26"), "yolov8"]
    last_error: Exception | None = None
    for fmt in dict.fromkeys(formats):
        try:
            version.download(model_format=fmt, location=str(out_dir), overwrite=True)
            break
        except Exception as exc:  # SDK/export compatibility fallback
            last_error = exc
    else:
        raise RuntimeError(f"Roboflow download failed: {last_error}")

    data_yaml = find_data_yaml(out_dir)
    normalize_data_yaml(data_yaml)
    manifest = {
        "dataset": dcfg["name"],
        "workspace": dcfg["workspace"],
        "project": dcfg["project"],
        "version": dcfg["version"],
        "source_url": dcfg["source_url"],
        "data_yaml": str(data_yaml),
    }
    json_dump(manifest, "artifacts/dataset_manifest.json")
    return data_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    path = download_dataset(args.config, args.force)
    print(path)


if __name__ == "__main__":
    main()
