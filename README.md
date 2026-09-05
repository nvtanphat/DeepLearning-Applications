# 🚀 DeepLearning-Applications

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep_Learning-Applications-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
</p>

Kho lưu trữ các bài thực hành, nghiên cứu thực nghiệm và đồ án môn học **Học sâu và Ứng dụng (Deep Learning & Applications)**. Mỗi bài toán được tổ chức thành một dự án độc lập với quy trình hoàn chỉnh từ chuẩn bị dữ liệu, huấn luyện mô hình đến đánh giá thực nghiệm.

---

## 📂 Danh mục bài toán & Lộ trình thực hành (Roadmap)

| Ngày / Chuyên đề | Dự án | Mô tả bài toán & Điểm nhấn kỹ thuật | Trạng thái |
|:---:|---|---|:---:|
| **Day 1** | [**Vietnam Vehicle Detection — YOLO26 vs RT-DETR-L**](Day1/vn-traffic-yolo26-vs-rtdetr) | **Vietnam Vehicle Detection**: Đối đầu thực nghiệm phát hiện phương tiện giao thông giữa họ YOLO26 (Nano/Small/Medium) và RT-DETR-L trên điều kiện thực tế Việt Nam. Đo đạc đa chiều Trade-off (mAP, FPS, VRAM, Params). | <kbd>✅ Hoàn thành</kbd> |
| **Day 2** | *Đang cập nhật...* | *Bài toán tiếp theo trong chuỗi Deep Learning Applications.* | <kbd>⏳ Sắp ra mắt</kbd> |

---

## 🌟 Điểm nhấn kết quả nổi bật — Day 1 Benchmark

Đối đầu 4 mô hình trên tập **Test Split độc lập (740 ảnh phương tiện giao thông Việt Nam)** chạy trên GPU Nvidia Tesla T4:

| Mô hình | Phân khúc | Params | Checkpoint | VRAM Peak | Tốc độ (FPS) | mAP@50 | mAP@50:95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⚡ **YOLO26n** | Edge AI / Embedded | **2.51 M** | **5.14 MB** | **156 MB** | **94.62 FPS** | `52.57%` | `40.15%` |
| 🚀 **YOLO26s** | Real-time CCTV | 9.95 M | 19.38 MB | 376 MB | 92.26 FPS | `53.35%` | `40.18%` |
| 🏋️ **YOLO26m** | Balanced Accuracy | 21.79 M | 42.00 MB | 728 MB | 38.55 FPS | `53.50%` | `40.13%` |
| 🛡️ **RT-DETR-L** | Transformer Server | 32.82 M | 63.18 MB | 1034 MB | 21.09 FPS | **`56.32%`** | **`43.76%`** |

> 📌 *Xem toàn bộ biểu đồ đường cong huấn luyện, ma trận nhầm lẫn, phân tích 8 lớp phương tiện và hướng dẫn triển khai tại:*  
> 👉 [**Day 1 Full Documentation & Benchmark Report 📑**](Day1/vn-traffic-yolo26-vs-rtdetr/README.md)

---

## 🚀 Hướng dẫn bắt đầu nhanh

### 1. Clone repository
```bash
git clone https://github.com/nvtanphat/DeepLearning-Applications.git
cd DeepLearning-Applications
```

### 2. Chạy thử nghiệm từng dự án
Mỗi thư mục `DayX/` là một dự án Python độc lập:

```powershell
# Di chuyển vào dự án Day 1: Vietnam Vehicle Detection
cd Day1/vn-traffic-yolo26-vs-rtdetr

# Khởi tạo môi trường & cài đặt thư viện
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Chạy thử smoke-test 1 epoch
python -m src.run_all --config configs/quick.yaml
```

---

## 📜 Giấy phép (License)
Dự án được phát hành theo giấy phép [MIT License](Day1/vn-traffic-yolo26-vs-rtdetr/LICENSE). Mọi đóng góp và trích dẫn vui lòng giữ nguyên bản quyền nguồn.
