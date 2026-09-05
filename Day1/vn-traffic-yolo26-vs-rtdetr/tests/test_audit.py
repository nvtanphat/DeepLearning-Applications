from pathlib import Path

from src.audit_dataset import audit_yolo_labels


def test_audit_yolo_labels(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    (labels / "b.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = audit_yolo_labels(labels, num_classes=2)
    assert report["boxes"] == 2
    assert report["invalid_count"] == 1
    assert report["class_counts"] == {0: 1, 1: 1}
