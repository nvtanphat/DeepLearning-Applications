# DeepLearning-Applications

Kho lưu trữ các bài thực hành, nghiên cứu thực nghiệm và đồ án ứng dụng **Deep Learning & Computer Vision** (Học sâu và ứng dụng).

---

## 📂 Danh mục bài toán & Dự án

| Ngày / Chuyên đề | Thư mục | Mô tả bài toán | Công nghệ / Mô hình | Kết quả tiêu biểu |
|---|---|---|---|---|
| **Day 1** | [Day1/vn-traffic-yolo26-vs-rtdetr](Day1/vn-traffic-yolo26-vs-rtdetr) | **Vietnam Traffic Benchmark**: So sánh thực nghiệm phát hiện phương tiện giao thông tại Việt Nam | **YOLO26 Variants (Nano/Small/Medium) vs RT-DETR-L**, PyTorch, Ultralytics | YOLO26s: **92.3 FPS**, 19 MB; RT-DETR: **43.8% mAP50-95** |

---

### 📊 Điểm nhấn Benchmark Day 1: Trade-off giữa YOLO26s và RT-DETR-L

<p align="center">
  <img src="Day1/vn-traffic-yolo26-vs-rtdetr/docs/images/overall_metrics.png" width="48%" />
  <img src="Day1/vn-traffic-yolo26-vs-rtdetr/docs/images/accuracy_speed_tradeoff.png" width="48%" />
</p>

| Model | mAP50-95 | mAP50 | Precision | Recall | FPS (Batch=1) | Peak VRAM | Params |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26s** | `0.4018` | `0.5335` | **`0.7852`** | `0.5191` | **`92.26 FPS`** | **`376 MB`** | **`9.95 M`** |
| **RT-DETR-L** | **`0.4376`** | **`0.5632`** | `0.7313` | **`0.5663`** | `21.09 FPS` | `1034 MB` | `32.82 M` |

---

## 🚀 Hướng dẫn bắt đầu nhanh

### 1. Clone repository về máy
```bash
git clone https://github.com/nvtanphat/DeepLearning-Applications.git
cd DeepLearning-Applications
```

### 2. Khám phá từng bài toán cụ thể
Mỗi bài toán trong thư mục `DayX/` là một dự án hoàn chỉnh độc lập, có tài liệu hướng dẫn và môi trường riêng:

- Đi tới [Day 1: VN Traffic YOLO26 vs RT-DETR](Day1/vn-traffic-yolo26-vs-rtdetr):
  ```powershell
  cd Day1/vn-traffic-yolo26-vs-rtdetr
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python -m src.run_all --config configs/quick.yaml
  ```

---

## 📜 Giấy phép (License)
Dự án được phát hành theo giấy phép [MIT License](Day1/vn-traffic-yolo26-vs-rtdetr/LICENSE).
