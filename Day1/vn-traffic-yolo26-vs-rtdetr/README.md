# 🇻🇳 Vietnam Traffic Benchmark — YOLO26 Variants (Nano/Small/Medium) vs RT-DETR-L

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.4.139-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset: Roboflow](https://img.shields.io/badge/dataset-Roboflow%20Universe-purple.svg)](https://universe.roboflow.com/jinkun1998s-workspace/vietnam-vehicle-detection-nt45b)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dự án nghiên cứu và benchmark thực nghiệm end-to-end so sánh giữa các biến thể thuộc họ **YOLO26** (kiến trúc CNN one-stage mới nhất) với **RT-DETR-L** (Real-Time Detection Transformer):
- **YOLO26n (Nano)**: Tối ưu cho thiết bị nhúng / Edge AI, FPS tối đa, tiết kiệm VRAM.
- **YOLO26s (Small)**: Mô hình chuẩn cân bằng hài hòa giữa độ chính xác và tốc độ.
- **YOLO26m (Medium)**: Dung lượng mô hình lớn hơn, tăng cường độ chính xác nhận diện.
- **RT-DETR-L (Large)**: Đại diện cho kiến trúc Hybrid Transformer loại bỏ hoàn toàn NMS.

Thực nghiệm được thực hiện trên tập dữ liệu đặc thù giao thông đường bộ Việt Nam với mật độ phương tiện hỗn hợp cao (xe máy, xe đạp, ô tô, xe buýt, xe tải,...). Dự án được thiết kế **chạy trực tiếp tại Local/Server** độc lập, tự động hóa từ tải dữ liệu, kiểm tra nhãn, huấn luyện, đánh giá định lượng đến xuất báo cáo trực quan.

---

## 📑 Mục lục
1. [Bộ dữ liệu (Dataset)](#1-bộ-dữ-liệu-dataset)
2. [Thiết kế thực nghiệm & Fair-Comparison Protocol](#2-thiết-kế-thực-nghiệm--fair-comparison-protocol)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Hướng dẫn chạy thực nghiệm Local](#5-hướng-dẫn-chạy-thực-nghiệm-local)
6. [Pipeline đánh giá, Benchmark & Chẩn đoán](#6-pipeline-đánh-giá-benchmark--chẩn-đoán)
7. [Inference & Export mô hình](#7-inference--export-mô-hình)
8. [Kiểm thử tự động (Unit Tests)](#8-kiểm-thử-tự-động-unit-tests)
9. [Sản phẩm đầu ra (Artifacts & Reports)](#9-sản-phẩm-đầu-ra-artifacts--reports)
10. [Giấy phép (License)](#10-giấy-phép-license)

---

## 1. Bộ dữ liệu (Dataset)

**Vietnam Vehicle Detection v1** (Nguồn: [Roboflow Universe](https://universe.roboflow.com/jinkun1998s-workspace/vietnam-vehicle-detection-nt45b/dataset/1))

- **Quy mô**: 3,376 ảnh chụp thực tế điều kiện giao thông đường bộ tại Việt Nam.
- **Phân chia split cố định**:
  - `train`: 2,062 ảnh (~61.1%)
  - `valid`: 574 ảnh (~17.0%)
  - `test`: 740 ảnh (~21.9%)
- **Số lớp đối tượng (8 classes)**:
  1. `xe buyt` (Bus)
  2. `xe container` (Container Truck)
  3. `xe cuu hoa` (Fire Truck)
  4. `xe dap` (Bicycle)
  5. `xe hoi` (Car)
  6. `xe may` (Motorbike)
  7. `xe tai` (Truck)
  8. `xe van` (Van)
- **Định dạng nhãn**: YOLO standard format (`<class_id> <x_center> <y_center> <width> <height>`).
- **Bản quyền dataset**: CC BY 4.0.

> [!NOTE]
> Dataset được tải tự động về máy thông qua Roboflow API Key được khai báo trong file `.env`.

---

## 2. Thiết kế thực nghiệm & Fair-Comparison Protocol

So sánh đa chiều theo bài toán đánh đổi (**Trade-off analysis**):

$$\text{Accuracy (mAP, F1, Per-Class AP)} \iff \text{Latency / FPS} \iff \text{VRAM Footprint} \iff \text{Model Size / Params}$$

| Tiêu chí | YOLO26n | YOLO26s | YOLO26m | RT-DETR-L |
|---|---|---|---|---|
| **Họ kiến trúc** | Advanced CNN (One-stage) | Advanced CNN (One-stage) | Advanced CNN (One-stage) | Hybrid Transformer (NMS-free) |
| **Checkpoint gốc** | `yolo26n.pt` | `yolo26s.pt` | `yolo26m.pt` | `rtdetr-l.pt` |
| **Mục tiêu tối ưu** | Edge AI / Max FPS | Cân bằng Speed/mAP | Nâng cao Accuracy | Transformer SOTA |
| **Input resolution**| $640 \times 640$ | $640 \times 640$ | $640 \times 640$ | $640 \times 640$ |
| **Số epoch** | 50 (patience = 15) | 50 (patience = 15) | 50 (patience = 15) | 50 (patience = 15) |
| **Batch size** | 4 | 4 | 4 | 4 |
| **Random Seed** | 42 | 42 | 42 | 42 |
| **Data Augmentation**| Giống nhau | Giống nhau | Giống nhau | Giống nhau |
| **AMP (Mixed Prec.)**| `True` | `True` | `True` | `False` |
| **Deterministic** | `True` | `True` | `True` | `False` |

---

## 3. Cấu trúc thư mục dự án

```text
vn-traffic-yolo26-vs-rtdetr/
├── configs/
│   ├── experiment.yaml          # Cấu hình thực nghiệm chuẩn (50 epochs, 4 models, full test)
│   └── quick.yaml               # Cấu hình smoke-test nhanh (1 epoch, sanity check)
├── src/
│   ├── common.py                # Hàm load config, đường dẫn, device, environment logging
│   ├── download_data.py         # Tự động tải dữ liệu từ Roboflow Universe
│   ├── audit_dataset.py         # Kiểm tra tính hợp lệ của nhãn & vẽ phân bố dataset
│   ├── train.py                 # Huấn luyện bất kỳ model nào hoặc tất cả với early stopping
│   ├── evaluate.py              # Đánh giá mAP50, mAP50-95 và per-class metrics trên test split
│   ├── benchmark.py             # Đo latency (mean/p50/p95), FPS, VRAM peak, RAM footprint
│   ├── visualize_predictions.py # Sinh ảnh trực quan so sánh GT vs Models & worst cases
│   ├── report.py                # Tổng hợp toàn bộ số liệu ra REPORT.md và biểu đồ so sánh đa mô hình
│   ├── run_all.py               # Chạy pipeline toàn diện từ A đến Z tại local
│   ├── infer.py                 # Nhận diện đối tượng trên ảnh hoặc video tùy chọn
│   └── export_model.py          # Xuất mô hình sang ONNX / TensorRT / OpenVINO
├── tests/                       # Unit tests kiểm tra config, audit contracts và pipeline
├── artifacts/                   # Thư mục chứa báo cáo, biểu đồ, kết quả xuất
├── runs/                        # Checkpoints và log training của Ultralytics
├── .env.example                 # Mẫu cấu hình API key
├── pytest.ini                   # Cấu hình pytest
├── requirements.txt             # Thư viện phục vụ chạy thực nghiệm
├── requirements-dev.txt         # Thư viện cho kiểm thử và development
└── README.md                    # Tài liệu hướng dẫn dự án
```

---

## 4. Cài đặt môi trường

### 4.1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 hoặc Linux (Ubuntu 20.04+).
- **Python**: 3.10 hoặc 3.11.
- **GPU**: Khuyến nghị có GPU Nvidia (CUDA 11.8 hoặc 12.x) tối thiểu 6GB-8GB VRAM (hoặc chạy CPU chế độ test).

### 4.2. Cài đặt chi tiết

```powershell
# 1. Di chuyển vào thư mục dự án
cd Day1/vn-traffic-yolo26-vs-rtdetr

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

# 3. Cài đặt dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Cấu hình Roboflow API Key
Copy-Item .env.example .env    # Linux/macOS: cp .env.example .env
# Mở file .env và điền: ROBOFLOW_API_KEY=your_key_here
```

---

## 5. Hướng dẫn chạy thực nghiệm Local

### 5.1. Chạy nhanh kiểm tra luồng (Smoke Test)
Sử dụng cấu hình rút gọn `quick.yaml` (1 epoch, tập mẫu nhỏ) để xác nhận hệ thống hoạt động chính xác:
```powershell
python -m src.run_all --config configs/quick.yaml
```

### 5.2. Chạy toàn bộ Pipeline chuẩn (Full Run)
Lệnh tự động chạy toàn bộ quy trình: Tải dữ liệu $\to$ Audit nhãn $\to$ Train tất cả models $\to$ Đánh giá Test set $\to$ Benchmark FPS/VRAM $\to$ Ghép ảnh trực quan $\to$ Xuất báo cáo tổng hợp:
```powershell
python -m src.run_all --config configs/experiment.yaml
```

### 5.3. Chạy từng module độc lập

```powershell
# 1. Tải dữ liệu từ Roboflow
python -m src.download_data --config configs/experiment.yaml

# 2. Audit dữ liệu & trực quan hóa phân bố nhãn
python -m src.audit_dataset --config configs/experiment.yaml

# 3. Huấn luyện riêng từng model (yolo26n, yolo26, yolo26m, rtdetr, hoặc all)
python -m src.train --model yolo26n --config configs/experiment.yaml
python -m src.train --model yolo26  --config configs/experiment.yaml
python -m src.train --model yolo26m --config configs/experiment.yaml
python -m src.train --model rtdetr  --config configs/experiment.yaml
python -m src.train --model all     --config configs/experiment.yaml

# 4. Resume training nếu bị gián đoạn
python -m src.train --model yolo26m --resume runs\yolo26m_vn_traffic\weights\last.pt

# 5. Đánh giá test split (hỗ trợ từng model hoặc all)
python -m src.evaluate --model all --config configs/experiment.yaml

# 6. Đo lường tốc độ & VRAM benchmark
python -m src.benchmark --model all --config configs/experiment.yaml

# 7. Trực quan hóa ảnh dự đoán & phân tích lỗi (so sánh đa mô hình)
python -m src.visualize_predictions --config configs/experiment.yaml

# 8. Sinh báo cáo tổng hợp REPORT.md
python -m src.report --config configs/experiment.yaml
```

---

## 6. Pipeline đánh giá, Benchmark & Chẩn đoán

### 6.1. Đánh giá độ chính xác (Accuracy Metrics)
- Thực hiện trên **tập test độc lập** (740 ảnh).
- Cấu hình đánh giá: `conf=0.001`, `iou=0.70` quét toàn bộ đường cong Precision-Recall.
- Xuất đầy đủ: **mAP50-95** (chỉ số chính), **mAP50**, **mAP75**, **Precision**, **Recall**, **F1-score** theo từng lớp.

### 6.2. Benchmark tốc độ & Bộ nhớ (Deployment Benchmark)
- Batch size = 1, kích thước $640 \times 640$.
- Warmup 20 ảnh, đối chuẩn trên 200 ảnh test nạp sẵn vào RAM.
- Đồng bộ `torch.cuda.synchronize()` trước và sau khi inference để đo độ trễ thực của GPU.
- Đo lường: **Mean / Median / p95 Latency**, **FPS**, **CUDA Peak Allocated/Reserved VRAM**, **Model Parameters**, **Checkpoint Size**.

### 6.3. Phân tích trực quan & Chẩn đoán ca khó (Error Analysis)
- Ghép ảnh so sánh đồng thời các góc nhìn (Side-by-side): **Ground Truth** $\leftrightarrow$ **Tất cả Models** (`conf=0.25`, `iou=0.50`).
- Tự động lọc và lưu các ảnh có điểm F1 thấp nhất vào `worst_<model>/` phục vụ mổ xẻ các trường hợp nhận diện sai hoặc bỏ sót của từng mô hình.

---

## 7. Inference & Export mô hình

### 7.1. Nhận diện trên ảnh & video tùy biến
```powershell
# Nhận diện trên ảnh với biến thể mong muốn
python -m src.infer --source test_image.jpg --model yolo26n
python -m src.infer --source test_image.jpg --model yolo26m
python -m src.infer --source test_image.jpg --model rtdetr

# Nhận diện trên video giao thông
python -m src.infer --source traffic_video.mp4 --model yolo26n --conf 0.3
```

### 7.2. Xuất mô hình phục vụ triển khai
```powershell
# Xuất định dạng ONNX
python -m src.export_model --model yolo26n --format onnx
python -m src.export_model --model yolo26m --format onnx
python -m src.export_model --model rtdetr  --format onnx

# Xuất TensorRT (yêu cầu GPU NVIDIA + TensorRT)
python -m src.export_model --model yolo26n --format engine
```

---

## 8. Kiểm thử tự động (Unit Tests)

Chạy bộ kiểm thử với `pytest`:
```powershell
python -m pip install -r requirements-dev.txt
pytest -v
```

---

## 9. Sản phẩm đầu ra (Artifacts & Reports)

Toàn bộ kết quả sau khi chạy được tập hợp tại `artifacts/reports/`:

```text
artifacts/reports/
├── REPORT.md                         # Báo cáo tổng kết toàn diện
├── environment.json                  # Chi tiết phần cứng & môi trường chạy
├── comparison.csv / comparison.md    # Bảng so sánh đối đầu định lượng giữa tất cả models
├── per_class_comparison.csv          # Chi tiết mAP50, mAP75, mAP50-95 theo 8 lớp của từng model
├── dataset/                          # Biểu đồ phân bổ dữ liệu
├── figures/                          # Biểu đồ so sánh metric, FPS, VRAM
├── val_runs/                         # Đường cong PR, F1, ma trận nhầm lẫn của từng model
└── qualitative/                      # Ảnh side-by-side & failure cases theo từng model
```

---

## 10. Giấy phép (License)

- **Mã nguồn**: Phát hành theo giấy phép [MIT License](LICENSE).
- **Dữ liệu**: Nguồn Roboflow Universe, áp dụng theo điều khoản [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
