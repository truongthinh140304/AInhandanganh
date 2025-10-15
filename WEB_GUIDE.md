# 🌐 Hướng dẫn sử dụng ứng dụng web AI Object Recognition

## 🚀 Khởi động ứng dụng

### Cách 1: Khởi động nhanh
```bash
python start_web.py
```
Script này sẽ tự động mở trình duyệt tại `http://localhost:5000`

### Cách 2: Khởi động thủ công
```bash
python app.py
```
Sau đó truy cập: `http://localhost:5000`

## 📱 Giao diện web

### 1. **Trang chủ** (`/`)
- **Upload ảnh**: Kéo thả hoặc click để chọn ảnh
- **Hỗ trợ định dạng**: JPG, PNG, JPEG, GIF, BMP
- **Kích thước tối đa**: 16MB
- **Tính năng**: Drag & drop, preview ảnh

### 2. **Trang kết quả** (`/results/<filename>`)
- **Ảnh gốc**: Hiển thị ảnh đã upload
- **Ảnh kết quả**: Ảnh với bounding boxes (nếu có)
- **Thống kê**: Số người, vật dụng, độ tin cậy
- **Phân tích chi tiết**: Giới tính, màu áo, vật dụng từng người
- **Phong cảnh**: Thông tin về môi trường

### 3. **Trang lịch sử** (`/history`)
- **Xem lại**: Tất cả ảnh đã phân tích
- **Thống kê nhanh**: Số người, vật dụng, độ tin cậy
- **Xem chi tiết**: Click vào ảnh để xem kết quả đầy đủ

## 🎯 Cách sử dụng

### Bước 1: Upload ảnh
1. Truy cập `http://localhost:5000`
2. Kéo thả ảnh vào vùng upload hoặc click "Chọn ảnh"
3. Chọn file ảnh từ máy tính
4. Click "Phân tích ảnh"

### Bước 2: Xem kết quả
1. Ứng dụng sẽ tự động chuyển đến trang kết quả
2. Xem thống kê tổng quan
3. Xem phân tích chi tiết từng người
4. Xem thông tin phong cảnh

### Bước 3: Lưu trữ và xem lại
1. Kết quả được lưu tự động
2. Truy cập "Lịch sử" để xem lại
3. Click vào ảnh để xem chi tiết

## 📊 Thông tin hiển thị

### Thống kê tổng quan:
- **Số người**: Tổng số người phát hiện được
- **Số vật dụng**: Tổng số vật dụng phát hiện được
- **Độ tin cậy**: Độ tin cậy của phân tích phong cảnh
- **Đã phân tích**: Số người đã được phân tích chi tiết

### Phân tích từng người:
- **Giới tính**: Nam/Nữ (độ tin cậy ~70-80%)
- **Màu áo**: Màu sắc trang phục
- **Vật dụng**: Điện thoại, túi xách, balo, v.v.
- **Phong cảnh**: Thông tin môi trường

### Phong cảnh:
- **Trong nhà**: Môi trường indoor
- **Ngoài trời**: Môi trường outdoor
- **Thời tiết**: Nắng, mưa, u ám, v.v.

## 🔧 Tùy chỉnh

### Thay đổi cổng:
Trong `app.py`, dòng cuối:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```
Thay đổi `port=5000` thành cổng khác.

### Thay đổi kích thước file tối đa:
Trong `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### Thay đổi thư mục upload:
Trong `app.py`:
```python
UPLOAD_FOLDER = 'data/uploads'
```

## 🐛 Xử lý lỗi

### Lỗi không khởi động được:
```bash
# Kiểm tra port đã được sử dụng
netstat -an | findstr :5000

# Thay đổi port
python app.py
# Sau đó sửa port trong app.py
```

### Lỗi upload file:
- Kiểm tra định dạng file (JPG, PNG, JPEG, GIF, BMP)
- Kiểm tra kích thước file (< 16MB)
- Kiểm tra quyền ghi trong thư mục `data/uploads`

### Lỗi phân tích ảnh:
- Kiểm tra ảnh có chứa người không
- Thử với ảnh khác
- Kiểm tra log trong terminal

## 📱 Responsive Design

Ứng dụng web được thiết kế responsive:
- **Desktop**: Giao diện đầy đủ
- **Tablet**: Tối ưu cho màn hình trung bình
- **Mobile**: Giao diện thân thiện với điện thoại

## 🔒 Bảo mật

### Development Mode:
- Debug mode: Bật (chỉ dùng cho development)
- Secret key: Cần thay đổi cho production

### Production Mode:
```python
# Trong app.py
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🚀 Deploy lên production

### 1. Sử dụng Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 2. Sử dụng Docker:
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### 3. Deploy lên cloud:
- **Heroku**: Sử dụng Procfile
- **AWS**: Sử dụng Elastic Beanstalk
- **Google Cloud**: Sử dụng App Engine
- **Azure**: Sử dụng App Service

## 📈 Monitoring

### Log files:
- Ứng dụng ghi log vào terminal
- Có thể redirect vào file:
```bash
python app.py > app.log 2>&1
```

### Performance:
- Thời gian xử lý: ~2-5 giây/ảnh (CPU)
- Memory usage: ~500MB-1GB
- Storage: ~100MB cho models

## 🎉 Kết luận

Ứng dụng web AI Object Recognition đã sẵn sàng sử dụng!

**Tính năng chính:**
- ✅ Upload ảnh qua giao diện web
- ✅ Phân tích AI tự động
- ✅ Hiển thị kết quả trực quan
- ✅ Lưu trữ lịch sử
- ✅ Responsive design
- ✅ API endpoint

**Cách sử dụng:**
1. Chạy `python start_web.py`
2. Upload ảnh có chứa người
3. Xem kết quả phân tích
4. Lưu trữ và xem lại

**Chúc bạn sử dụng hiệu quả! 🚀**
