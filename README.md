# DeepLearning-Applications

Kho lưu trữ các bài thực hành, nghiên cứu thực nghiệm và đồ án ứng dụng **Deep Learning & Computer Vision** (Học sâu và ứng dụng).

---

## 📂 Danh mục bài toán & Dự án

| Ngày / Chuyên đề | Thư mục | Mô tả bài toán | Công nghệ / Mô hình |
|---|---|---|---|
| **Day 1** | [Day1/vn-traffic-yolo26-vs-rtdetr](Day1/vn-traffic-yolo26-vs-rtdetr) | **Vietnam Traffic Benchmark**: So sánh thực nghiệm toàn diện phát hiện phương tiện giao thông tại Việt Nam | **YOLO26 Variants (Nano/Small/Medium) vs RT-DETR-L**, PyTorch, Ultralytics, Roboflow |

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
