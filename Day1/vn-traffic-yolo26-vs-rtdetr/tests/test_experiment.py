import json
from pathlib import Path

from src.common import load_config


def test_fair_budget_is_shared():
    cfg = load_config("configs/experiment.yaml")
    exp = cfg["experiment"]
    assert exp["imgsz"] == 640
    assert exp["batch"] == 4
    assert exp["seed"] == 42
    assert cfg["models"]["yolo26"]["checkpoint"] == "yolo26s.pt"
    assert cfg["models"]["rtdetr"]["checkpoint"] == "rtdetr-l.pt"


def test_kaggle_template_uses_t4_and_internet():
    meta = json.loads(Path("kaggle/kernel-metadata.template.json").read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
