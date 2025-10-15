import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys

def detect_dominant_color(image_region):
    """
    Phát hiện màu chủ đạo trong vùng ảnh
    """
    # Chuyển đổi từ BGR sang RGB
    image_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
    
    # Reshape ảnh thành mảng 2D
    pixels = image_rgb.reshape(-1, 3)
    
    # Sử dụng K-means để tìm màu chủ đạo
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Lấy màu chủ đạo
    dominant_color = kmeans.cluster_centers_[0]
    
    # Chuyển đổi RGB sang tên màu
    color_name = rgb_to_color_name(dominant_color)
    
    return color_name, dominant_color

def rgb_to_color_name(rgb):
    """
    Chuyển đổi RGB sang tên màu tiếng Việt
    """
    r, g, b = rgb
    
    # Chuyển sang HSV để dễ phân loại màu
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h = h * 360
    s = s * 100
    v = v * 100
    
    # Phân loại màu dựa trên hue và saturation
    if v < 30:
        return "Đen"
    elif v > 80 and s < 20:
        return "Trắng"
    elif s < 20:
        return "Xám"
    elif 0 <= h <= 15 or 345 <= h <= 360:
        return "Đỏ"
    elif 15 < h <= 45:
        return "Cam"
    elif 45 < h <= 75:
        return "Vàng"
    elif 75 < h <= 165:
        return "Xanh lá"
    elif 165 < h <= 195:
        return "Xanh dương"
    elif 195 < h <= 285:
        return "Tím"
    elif 285 < h <= 345:
        return "Hồng"
    else:
        return "Nâu"

def extract_person_region(image, bbox):
    """
    Trích xuất vùng người từ bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    person_region = image[y1:y2, x1:x2]
    return person_region

def extract_clothing_region(image, bbox):
    """
    Trích xuất vùng áo từ phần thân trên của người
    """
    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    
    # Lấy phần thân trên (từ đầu đến giữa người)
    clothing_y1 = y1 + int(height * 0.1)  # Bắt đầu từ cổ
    clothing_y2 = y1 + int(height * 0.6)  # Kết thúc ở giữa người
    
    clothing_region = image[clothing_y1:clothing_y2, x1:x2]
    return clothing_region

def draw_detection_results(image, detections, colors):
    """
    Vẽ kết quả phát hiện lên ảnh
    """
    result_image = image.copy()
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[i % len(colors)]
        
        # Vẽ bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label và confidence
        label_text = f"{label}: {confidence:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Vẽ background cho text
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Vẽ text
        cv2.putText(result_image, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image

def create_results_table(results):
    """
    Tạo bảng kết quả dạng text
    """
    if not results:
        return "Không phát hiện người nào trong ảnh."
    
    table = "=" * 80 + "\n"
    table += f"{'Người':<8} {'Giới tính':<12} {'Màu áo':<12} {'Vật dụng':<20} {'Cảnh vật':<15}\n"
    table += "=" * 80 + "\n"
    
    for i, result in enumerate(results, 1):
        person = str(i)
        gender = result.get('gender', 'Không xác định')
        clothing_color = result.get('clothing_color', 'Không xác định')
        objects = ', '.join(result.get('objects', [])) if result.get('objects') else 'Không có'
        scene = result.get('scene', 'Không xác định')
        
        table += f"{person:<8} {gender:<12} {clothing_color:<12} {objects:<20} {scene:<15}\n"
    
    table += "=" * 80
    return table

def save_results_image(image, output_path):
    """
    Lưu ảnh kết quả
    """
    cv2.imwrite(output_path, image)
    print(f"Ảnh kết quả đã được lưu tại: {output_path}")

def get_color_palette():
    """
    Trả về danh sách màu để vẽ bounding box
    """
    return [
        (0, 255, 0),    # Xanh lá
        (255, 0, 0),    # Xanh dương
        (0, 0, 255),    # Đỏ
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Vàng
        (128, 0, 128),  # Tím
        (255, 165, 0),  # Cam
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]

class SimpleGenderClassifier(nn.Module):
    """
    Model CNN đơn giản để phân loại giới tính
    """
    def __init__(self):
        super(SimpleGenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 classes: nam, nữ
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_gender_transform():
    """
    Trả về transform cho model phân loại giới tính
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def classify_gender_simple(image_region):
    """
    Phân loại giới tính đơn giản dựa trên đặc trưng hình học
    (Fallback method khi không có model được train)
    """
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return "Không xác định", 0.5
    
    # Lấy khuôn mặt lớn nhất
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Trích xuất vùng khuôn mặt
    face_region = image_region[y:y+h, x:x+w]
    
    # Phân tích đặc trưng đơn giản
    # (Trong thực tế, bạn nên train một model CNN thực sự)
    
    # Tính tỷ lệ chiều rộng/chiều cao khuôn mặt
    aspect_ratio = w / h
    
    # Tính độ sáng trung bình
    brightness = np.mean(gray[y:y+h, x:x+w])
    
    # Heuristic đơn giản (không chính xác, chỉ để demo)
    if aspect_ratio > 0.8 and brightness > 100:
        return "Nam", 0.6
    else:
        return "Nữ", 0.6

def classify_scene_simple(image):
    """
    Phân loại phong cảnh đơn giản dựa trên màu sắc và độ sáng
    """
    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính độ sáng trung bình
    brightness = np.mean(hsv[:, :, 2])
    
    # Tính độ bão hòa trung bình
    saturation = np.mean(hsv[:, :, 1])
    
    # Phân tích màu chủ đạo
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    
    blue_ratio = np.sum(blue_mask) / (image.shape[0] * image.shape[1])
    green_ratio = np.sum(green_mask) / (image.shape[0] * image.shape[1])
    
    # Heuristic phân loại
    if brightness < 80:
        return "Trong nhà", 0.7
    elif blue_ratio > 0.3:
        return "Ngoài trời - Trời xanh", 0.8
    elif green_ratio > 0.3:
        return "Ngoài trời - Cây xanh", 0.8
    elif brightness > 150:
        return "Ngoài trời - Nắng", 0.7
    else:
        return "Ngoài trời - U ám", 0.6
