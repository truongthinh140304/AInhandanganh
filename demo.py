#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script cho AI Object Recognition Application
"""

import os
import cv2
import numpy as np

def create_demo_image():
    """
    Tạo ảnh demo đơn giản
    """
    # Tạo ảnh trắng 640x480
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Vẽ một hình chữ nhật đơn giản để giả lập người
    cv2.rectangle(image, (200, 100), (400, 400), (0, 0, 0), -1)  # Người đen
    cv2.rectangle(image, (250, 150), (350, 250), (255, 255, 255), -1)  # Áo trắng
    
    # Vẽ một hình tròn để giả lập vật dụng
    cv2.circle(image, (300, 350), 30, (0, 255, 0), -1)  # Vật dụng xanh
    
    # Lưu ảnh demo
    demo_path = "data/demo_image.jpg"
    cv2.imwrite(demo_path, image)
    print(f"Da tao anh demo tai: {demo_path}")
    
    return demo_path

def test_basic_imports():
    """
    Test import các thư viện cơ bản
    """
    print("Testing basic imports...")
    
    try:
        import cv2
        print("+ OpenCV imported successfully")
    except ImportError as e:
        print(f"- OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("+ NumPy imported successfully")
    except ImportError as e:
        print(f"- NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("+ Ultralytics imported successfully")
    except ImportError as e:
        print(f"- Ultralytics import failed: {e}")
        return False
    
    try:
        import torch
        print(f"+ PyTorch imported successfully (device: {'CUDA' if torch.cuda.is_available() else 'CPU'})")
    except ImportError as e:
        print(f"- PyTorch import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """
    Test tải model YOLO
    """
    print("\nTesting YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("+ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"- YOLO model loading failed: {e}")
        return False

def test_utils_functions():
    """
    Test các hàm utils
    """
    print("\nTesting utils functions...")
    
    try:
        from utils import detect_dominant_color, rgb_to_color_name
        
        # Test với ảnh màu đỏ
        red_image = np.zeros((100, 100, 3), dtype=np.uint8)
        red_image[:, :, 2] = 255  # Kênh đỏ
        
        color_name, color_rgb = detect_dominant_color(red_image)
        print(f"+ Color detection test: {color_name}")
        
        # Test chuyển đổi RGB
        test_color = np.array([255, 0, 0])
        color_name = rgb_to_color_name(test_color)
        print(f"+ RGB to color name test: {color_name}")
        
        return True
    except Exception as e:
        print(f"- Utils functions test failed: {e}")
        return False

def main():
    """
    Hàm main cho demo
    """
    print("="*60)
    print("AI OBJECT RECOGNITION - DEMO")
    print("="*60)
    
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # Test imports
    if not test_basic_imports():
        print("\n- Basic imports failed!")
        return 1
    
    # Test YOLO model
    if not test_yolo_model():
        print("\n- YOLO model test failed!")
        return 1
    
    # Test utils
    if not test_utils_functions():
        print("\n- Utils test failed!")
        return 1
    
    # Tạo ảnh demo
    demo_image_path = create_demo_image()
    
    print("\n" + "="*60)
    print("DEMO HOAN THANH!")
    print("="*60)
    print("Cac thu vien da duoc cai dat thanh cong.")
    print("Ban co the chay ung dung chinh voi:")
    print(f"python main.py --image {demo_image_path}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())
