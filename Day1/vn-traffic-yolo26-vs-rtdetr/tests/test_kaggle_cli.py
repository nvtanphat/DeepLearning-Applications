import json
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not Path('kaggle/run_kaggle.py').exists() or not Path('scripts/prepare_kaggle.py').exists(),
    reason="Kaggle files omitted in local-only distribution",
)


def test_kaggle_kernel_is_headless_no_user_secret_dependency():
    text = Path('kaggle/run_kaggle.py').read_text(encoding='utf-8')
    assert 'UserSecretsClient' not in text
    assert 'ROBOFLOW_API_KEY' not in text
    assert '/kaggle/input' in text
    assert '/tmp/vn_traffic_data' in text
    assert 'yolo26_last.pt' not in text  # constructed from key, avoids hard-coded single model path
    assert 'find_resume_checkpoint' in text


def test_prepare_kaggle_injects_private_dataset_source():
    subprocess.run(
        [sys.executable, 'scripts/prepare_kaggle.py', '--username', 'unit-test-user'],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads(Path('kaggle/kernel-metadata.json').read_text(encoding='utf-8'))
    assert meta['id'] == 'unit-test-user/vn-traffic-yolo26-vs-rt-detr'
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['enable_gpu'] is True
    assert meta['dataset_sources'] == ['unit-test-user/vn-traffic-vehicle-detection-v1']


def test_prepare_kaggle_can_attach_resume_dataset():
    subprocess.run(
        [
            sys.executable,
            'scripts/prepare_kaggle.py',
            '--username',
            'unit-test-user',
            '--resume-source',
            'unit-test-user/vn-traffic-yolo26-rtdetr-resume',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads(Path('kaggle/kernel-metadata.json').read_text(encoding='utf-8'))
    assert meta['dataset_sources'] == [
        'unit-test-user/vn-traffic-vehicle-detection-v1',
        'unit-test-user/vn-traffic-yolo26-rtdetr-resume',
    ]


def test_kernel_metadata_uses_json_booleans_and_report_pipeline():
    meta = json.loads(Path('kaggle/kernel-metadata.template.json').read_text(encoding='utf-8'))
    assert meta['is_private'] is True
    assert meta['enable_gpu'] is True
    assert meta['enable_internet'] is True
    text = Path('kaggle/run_kaggle.py').read_text(encoding='utf-8')
    for expected in ['BoxPR_curve.png', 'confusion_matrix_normalized.png', 'per_class_metrics.csv', 'REPORT.md', 'qualitative']:
        assert expected in text
