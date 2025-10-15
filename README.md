# 🤖 AI Object Recognition Application

Ứng dụng AI nhận dạng đối tượng trên ảnh sử dụng YOLOv8 và các thuật toán xử lý ảnh.

## 🎯 Tính năng

- **Nhận dạng người**: Phát hiện và đếm số lượng người trong ảnh
- **Phân loại giới tính**: Xác định giới tính (nam/nữ) của từng người
- **Nhận dạng màu áo**: Phân tích màu sắc trang phục
- **Phân loại phong cảnh**: Xác định môi trường (trong nhà/ngoài trời, thời tiết)
- **Nhận dạng vật dụng**: Phát hiện các vật dụng như điện thoại, túi xách, balo, v.v.

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA (tùy chọn, để tăng tốc GPU)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 🌐 Sử dụng giao diện web (Khuyến nghị)

```bash
# Khởi động ứng dụng web
python start_web.py
```

Truy cập: `http://localhost:5000`

### 💻 Sử dụng command line

```bash
python main.py --image data/test.jpg
```

### Các tham số

- `--image`: Đường dẫn đến ảnh cần phân tích (bắt buộc)
- `--model`: Đường dẫn đến model YOLO (mặc định: yolov8n.pt)
- `--output`: Thư mục lưu kết quả (mặc định: data/)

### Ví dụ

```bash
# Phân tích ảnh với model mặc định
python main.py --image data/sample.jpg

# Sử dụng model khác
python main.py --image data/sample.jpg --model yolov8s.pt

# Chỉ định thư mục output
python main.py --image data/sample.jpg --output results/
```

## 📁 Cấu trúc dự án

```
project/
├── app.py              # Ứng dụng web Flask
├── start_web.py        # Script khởi động web
├── main.py             # Ứng dụng command line
├── utils.py            # Các hàm hỗ trợ
├── requirements.txt    # Dependencies
├── README.md          # Hướng dẫn
├── WEB_GUIDE.md       # Hướng dẫn web
├── TRAINING_GUIDE.md  # Hướng dẫn training
├── QUICK_START.md     # Hướng dẫn nhanh
├── templates/         # HTML templates
├── static/           # CSS, JS, images
├── models/           # Thư mục chứa models
├── data/             # Thư mục chứa ảnh và kết quả
└── test/             # Thư mục test
```

## 📊 Kết quả

Ứng dụng sẽ hiển thị:

1. **Bảng kết quả chi tiết**:
   ```
   ================================================================================
   Người    Giới tính    Màu áo      Vật dụng             Cảnh vật        
   ================================================================================
   1        Nam          Đen         Điện thoại          Ngoài trời - Nắng
   2        Nữ           Trắng       Túi xách            Ngoài trời - Nắng
   ================================================================================
   ```

2. **Ảnh kết quả**: Ảnh gốc với các bounding box và nhãn
3. **Thống kê**: Số lượng người, vật dụng, thời gian xử lý

## 🔧 Cấu hình

### Model YOLO
- Mặc định sử dụng `yolov8n.pt` (nano - nhanh nhất)
- Có thể thay bằng `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` để tăng độ chính xác

### Điều chỉnh độ tin cậy
Trong `main.py`, dòng 85:
```python
results = self.yolo_model(image, conf=0.5, verbose=False)
```
Thay đổi `conf=0.5` để điều chỉnh ngưỡng độ tin cậy.

## 🎨 Tùy chỉnh

### Thêm đối tượng mới
Trong `main.py`, thêm vào `target_classes`:
```python
self.target_classes = {
    # ... existing classes ...
    80: 'new_object',  # Thêm đối tượng mới
}
```

### Thêm tên tiếng Việt
Trong `vietnamese_names`:
```python
self.vietnamese_names = {
    # ... existing names ...
    'new_object': 'Tên tiếng Việt',
}
```

## 🚨 Lưu ý

1. **Model giới tính**: Hiện tại sử dụng heuristic đơn giản. Để tăng độ chính xác, cần train model CNN riêng.

2. **Phân loại phong cảnh**: Sử dụng thuật toán đơn giản dựa trên màu sắc. Có thể cải thiện bằng model chuyên dụng.

3. **Hiệu suất**: 
   - CPU: ~2-5 giây/ảnh
   - GPU: ~0.5-1 giây/ảnh

## 🔮 Nâng cấp

### 1. Train model giới tính riêng
```python
# Sử dụng dataset như UTKFace hoặc CelebA
# Train CNN model với architecture phù hợp
```

### 2. Cải thiện phân loại phong cảnh
```python
# Sử dụng Places365 dataset
# Fine-tune ResNet hoặc ViT model
```

### 3. Thêm tính năng
- Nhận dạng cảm xúc
- Ước tính tuổi
- Phân loại trang phục chi tiết
- Nhận dạng hành động

## 🐛 Xử lý lỗi

### Lỗi import thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Lỗi CUDA
```bash
# Cài đặt PyTorch CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Lỗi model không tải
```bash
# Xóa cache và tải lại
rm -rf ~/.cache/ultralytics
python main.py --image data/test.jpg
```

## 📝 License

MIT License

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Liên hệ

Nếu có vấn đề hoặc góp ý, vui lòng tạo issue trên GitHub.

---

**Chúc bạn sử dụng ứng dụng hiệu quả! 🎉**