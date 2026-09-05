from src.common import load_config


def test_config_has_two_models():
    cfg = load_config("configs/experiment.yaml")
    assert set(cfg["models"]) == {"yolo26", "rtdetr"}
    assert cfg["dataset"]["version"] == 1
    assert cfg["experiment"]["imgsz"] == 640
