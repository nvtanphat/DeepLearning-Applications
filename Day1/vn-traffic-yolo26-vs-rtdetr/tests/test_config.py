from src.common import load_config


def test_config_has_models():
    cfg = load_config("configs/experiment.yaml")
    assert {"yolo26", "yolo26n", "yolo26m", "rtdetr"}.issubset(set(cfg["models"]))
    assert cfg["dataset"]["version"] == 1
    assert cfg["experiment"]["imgsz"] == 640
