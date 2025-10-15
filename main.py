#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Object Recognition Application
Ứng dụng AI nhận dạng đối tượng trên ảnh

Tác giả: AI Assistant
Ngày tạo: 2025
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

# Import các thư viện AI
try:
    from ultralytics import YOLO
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Loi import thu vien: {e}")
    print("Vui long cai dat cac thu vien can thiet:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Import các hàm từ utils
from utils import (
    detect_dominant_color, extract_person_region, extract_clothing_region,
    draw_detection_results, create_results_table, save_results_image,
    get_color_palette, classify_gender_simple, classify_scene_simple
)

class AIObjectRecognizer:
    """
    Lớp chính để nhận dạng đối tượng bằng AI
    """
    
    def __init__(self, model_path="yolov8n.pt"):
        """
        Khởi tạo AI Object Recognizer
        
        Args:
            model_path: Đường dẫn đến model YOLO
        """
        self.model_path = model_path
        self.yolo_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng device: {self.device}")
        
        # Danh sách các đối tượng quan tâm
        self.target_classes = {
            0: 'person',           # Người
            39: 'bottle',          # Chai
            41: 'cup',             # Cốc
            43: 'knife',           # Dao
            44: 'spoon',           # Thìa
            45: 'bowl',            # Bát
            46: 'banana',          # Chuối
            47: 'apple',           # Táo
            48: 'sandwich',        # Bánh sandwich
            49: 'orange',          # Cam
            50: 'broccoli',        # Bông cải
            51: 'carrot',          # Cà rốt
            52: 'hot dog',         # Xúc xích
            53: 'pizza',           # Pizza
            54: 'donut',           # Bánh donut
            55: 'cake',            # Bánh kem
            67: 'cell phone',      # Điện thoại
            68: 'book',            # Sách
            69: 'scissors',        # Kéo
            70: 'teddy bear',      # Gấu bông
            71: 'hair drier',      # Máy sấy tóc
            72: 'toothbrush',      # Bàn chải đánh răng
            73: 'backpack',        # Balo
            74: 'umbrella',        # Ô
            75: 'handbag',         # Túi xách
            76: 'tie',             # Cà vạt
            77: 'suitcase',        # Vali
        }
        
        # Mapping tên tiếng Việt
        self.vietnamese_names = {
            'person': 'Người',
            'cell phone': 'Điện thoại',
            'backpack': 'Balo',
            'umbrella': 'Ô',
            'handbag': 'Túi xách',
            'tie': 'Cà vạt',
            'suitcase': 'Vali',
            'book': 'Sách',
            'bottle': 'Chai',
            'cup': 'Cốc',
            'knife': 'Dao',
            'spoon': 'Thìa',
            'bowl': 'Bát',
        }
        
        self.load_model()
    
    def load_model(self):
        """
        Tải model YOLO
        """
        try:
            print("Đang tải model YOLO...")
            self.yolo_model = YOLO(self.model_path)
            print("Model YOLO đã được tải thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model YOLO: {e}")
            print("Model sẽ được tải tự động từ internet...")
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("Model YOLO đã được tải thành công!")
            except Exception as e2:
                print(f"Lỗi nghiêm trọng khi tải model: {e2}")
                sys.exit(1)
    
    def detect_objects(self, image):
        """
        Phát hiện đối tượng trong ảnh
        
        Args:
            image: Ảnh đầu vào (numpy array)
            
        Returns:
            List các detection results
        """
        try:
            # Chạy inference
            results = self.yolo_model(image, conf=0.5, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Lấy thông tin bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Chỉ quan tâm đến các class trong target_classes
                        if class_id in self.target_classes:
                            class_name = self.target_classes[class_id]
                            vietnamese_name = self.vietnamese_names.get(class_name, class_name)
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'class_name': class_name,
                                'vietnamese_name': vietnamese_name,
                                'confidence': confidence
                            })
            
            return detections
            
        except Exception as e:
            print(f"Lỗi khi phát hiện đối tượng: {e}")
            return []
    
    def analyze_person(self, image, person_bbox):
        """
        Phân tích thông tin của một người
        
        Args:
            image: Ảnh gốc
            person_bbox: Bounding box của người
            
        Returns:
            Dictionary chứa thông tin phân tích
        """
        try:
            # Trích xuất vùng người
            person_region = extract_person_region(image, person_bbox)
            
            # Phân tích giới tính
            gender, gender_confidence = classify_gender_simple(person_region)
            
            # Phân tích màu áo
            clothing_region = extract_clothing_region(image, person_bbox)
            if clothing_region.size > 0:
                clothing_color, _ = detect_dominant_color(clothing_region)
            else:
                clothing_color = "Không xác định"
            
            return {
                'gender': gender,
                'gender_confidence': gender_confidence,
                'clothing_color': clothing_color,
                'person_region': person_region,
                'clothing_region': clothing_region
            }
            
        except Exception as e:
            print(f"Lỗi khi phân tích người: {e}")
            return {
                'gender': 'Không xác định',
                'gender_confidence': 0.0,
                'clothing_color': 'Không xác định',
                'person_region': None,
                'clothing_region': None
            }
    
    def analyze_scene(self, image):
        """
        Phân tích phong cảnh/thời tiết
        
        Args:
            image: Ảnh gốc
            
        Returns:
            Tuple (scene_type, confidence)
        """
        try:
            return classify_scene_simple(image)
        except Exception as e:
            print(f"Lỗi khi phân tích phong cảnh: {e}")
            return "Không xác định", 0.0
    
    def process_image(self, image_path):
        """
        Xử lý ảnh và trả về kết quả phân tích
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            Dictionary chứa kết quả phân tích
        """
        print(f"Đang xử lý ảnh: {image_path}")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        print(f"Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
        
        # Phát hiện đối tượng
        print("Đang phát hiện đối tượng...")
        detections = self.detect_objects(image)
        print(f"Phát hiện được {len(detections)} đối tượng")
        
        # Phân tích phong cảnh
        print("Đang phân tích phong cảnh...")
        scene_type, scene_confidence = self.analyze_scene(image)
        
        # Phân tích từng người
        person_results = []
        person_detections = [d for d in detections if d['class_name'] == 'person']
        
        print(f"Phát hiện được {len(person_detections)} người")
        
        for i, person_detection in enumerate(person_detections):
            print(f"Đang phân tích người {i+1}...")
            
            # Phân tích thông tin người
            person_info = self.analyze_person(image, person_detection['bbox'])
            
            # Tìm các vật dụng gần người này
            person_objects = self.find_nearby_objects(person_detection, detections)
            
            person_result = {
                'person_id': i + 1,
                'bbox': person_detection['bbox'],
                'gender': person_info['gender'],
                'gender_confidence': person_info['gender_confidence'],
                'clothing_color': person_info['clothing_color'],
                'objects': person_objects,
                'scene': scene_type
            }
            
            person_results.append(person_result)
        
        # Tạo kết quả tổng hợp
        result = {
            'image_path': image_path,
            'image': image,
            'detections': detections,
            'person_results': person_results,
            'scene': scene_type,
            'scene_confidence': scene_confidence,
            'total_people': len(person_detections),
            'total_objects': len([d for d in detections if d['class_name'] != 'person'])
        }
        
        return result
    
    def find_nearby_objects(self, person_detection, all_detections):
        """
        Tìm các vật dụng gần người
        
        Args:
            person_detection: Detection của người
            all_detections: Tất cả detections
            
        Returns:
            List tên các vật dụng gần người
        """
        person_bbox = person_detection['bbox']
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        person_center_y = (person_bbox[1] + person_bbox[3]) / 2
        
        nearby_objects = []
        
        for detection in all_detections:
            if detection['class_name'] == 'person':
                continue
            
            obj_bbox = detection['bbox']
            obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
            obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2
            
            # Tính khoảng cách
            distance = np.sqrt((person_center_x - obj_center_x)**2 + 
                              (person_center_y - obj_center_y)**2)
            
            # Nếu vật dụng gần người (trong vòng 200 pixel)
            if distance < 200:
                nearby_objects.append(detection['vietnamese_name'])
        
        return nearby_objects
    
    def display_results(self, result):
        """
        Hiển thị kết quả phân tích
        
        Args:
            result: Kết quả từ process_image
        """
        print("\n" + "="*80)
        print("KẾT QUẢ PHÂN TÍCH ẢNH")
        print("="*80)
        
        # Thông tin tổng quan
        print(f"Ảnh: {result['image_path']}")
        print(f"Tổng số người: {result['total_people']}")
        print(f"Tổng số vật dụng: {result['total_objects']}")
        print(f"Phong cảnh: {result['scene']}")
        print()
        
        # Bảng kết quả chi tiết
        table = create_results_table(result['person_results'])
        print(table)
        
        # Hiển thị ảnh với kết quả
        self.show_image_with_results(result)
    
    def show_image_with_results(self, result):
        """
        Hiển thị ảnh với kết quả phát hiện
        
        Args:
            result: Kết quả từ process_image
        """
        try:
            image = result['image'].copy()
            colors = get_color_palette()
            
            # Vẽ bounding boxes
            for i, detection in enumerate(result['detections']):
                bbox = detection['bbox']
                label = detection['vietnamese_name']
                confidence = detection['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                color = colors[i % len(colors)]
                
                # Vẽ bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ label và confidence
                label_text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Vẽ background cho text
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Vẽ text
                cv2.putText(image, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Lưu ảnh kết quả
            output_path = "data/result_" + Path(result['image_path']).name
            save_results_image(image, output_path)
            
            # Hiển thị ảnh
            print(f"\nẢnh kết quả đã được lưu tại: {output_path}")
            
            # Hiển thị bằng matplotlib (tùy chọn)
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 8))
                plt.imshow(image_rgb)
                plt.title("Kết quả nhận dạng đối tượng")
                plt.axis('off')
                plt.tight_layout()
                
                # Lưu ảnh matplotlib
                plt.savefig("data/matplotlib_result.png", dpi=150, bbox_inches='tight')
                print("Ảnh matplotlib đã được lưu tại: data/matplotlib_result.png")
                
                # Hiển thị (uncomment để hiển thị popup)
                # plt.show()
                plt.close()
                
            except Exception as e:
                print(f"Lỗi khi hiển thị với matplotlib: {e}")
                
        except Exception as e:
            print(f"Lỗi khi hiển thị kết quả: {e}")

def main():
    """
    Hàm main của ứng dụng
    """
    parser = argparse.ArgumentParser(description='AI Object Recognition Application')
    parser.add_argument('--image', type=str, required=True, 
                       help='Đường dẫn đến ảnh cần phân tích')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Đường dẫn đến model YOLO (mặc định: yolov8n.pt)')
    parser.add_argument('--output', type=str, default='data/',
                       help='Thư mục lưu kết quả (mặc định: data/)')
    
    args = parser.parse_args()
    
    # Kiểm tra file ảnh
    if not os.path.exists(args.image):
        print(f"Lỗi: Không tìm thấy file ảnh: {args.image}")
        sys.exit(1)
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Khởi tạo AI Object Recognizer
        print("Khởi tạo AI Object Recognizer...")
        recognizer = AIObjectRecognizer(args.model)
        
        # Xử lý ảnh
        start_time = time.time()
        result = recognizer.process_image(args.image)
        end_time = time.time()
        
        # Hiển thị kết quả
        recognizer.display_results(result)
        
        print(f"\nThời gian xử lý: {end_time - start_time:.2f} giây")
        print("Hoàn thành!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
