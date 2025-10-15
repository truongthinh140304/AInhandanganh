#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script cho AI Object Recognition Application
"""

import os
import sys
import cv2
import numpy as np
from main import AIObjectRecognizer

def create_test_image():
    """
    Tạo ảnh test đơn giản
    """
    # Tạo ảnh trắng 640x480
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Vẽ một hình chữ nhật đơn giản để giả lập người
    cv2.rectangle(image, (200, 100), (400, 400), (0, 0, 0), -1)  # Người đen
    cv2.rectangle(image, (250, 150), (350, 250), (255, 255, 255), -1)  # Áo trắng
    
    # Vẽ một hình tròn để giả lập vật dụng
    cv2.circle(image, (300, 350), 30, (0, 255, 0), -1)  # Vật dụng xanh
    
    # Lưu ảnh test
    test_path = "data/test_image.jpg"
    cv2.imwrite(test_path, image)
    print(f"Đã tạo ảnh test tại: {test_path}")
    
    return test_path

def test_basic_functionality():
    """
    Test các chức năng cơ bản
    """
    print("="*60)
    print("TESTING AI OBJECT RECOGNITION APPLICATION")
    print("="*60)
    
    try:
        # Tạo ảnh test
        test_image_path = create_test_image()
        
        # Khởi tạo recognizer
        print("\n1. Khởi tạo AI Object Recognizer...")
        recognizer = AIObjectRecognizer()
        print("✓ Khởi tạo thành công!")
        
        # Test phân tích ảnh
        print("\n2. Phân tích ảnh test...")
        result = recognizer.process_image(test_image_path)
        print("✓ Phân tích thành công!")
        
        # Hiển thị kết quả
        print("\n3. Hiển thị kết quả...")
        recognizer.display_results(result)
        print("✓ Hiển thị thành công!")
        
        # Test các hàm utils
        print("\n4. Test các hàm utils...")
        test_utils_functions()
        print("✓ Test utils thành công!")
        
        print("\n" + "="*60)
        print("🎉 TẤT CẢ TEST ĐỀU THÀNH CÔNG!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_functions():
    """
    Test các hàm trong utils.py
    """
    from utils import detect_dominant_color, rgb_to_color_name, classify_scene_simple
    
    # Test tạo ảnh màu đỏ
    red_image = np.zeros((100, 100, 3), dtype=np.uint8)
    red_image[:, :, 2] = 255  # Kênh đỏ
    
    # Test phát hiện màu chủ đạo
    color_name, color_rgb = detect_dominant_color(red_image)
    print(f"   - Phát hiện màu đỏ: {color_name}")
    
    # Test phân loại phong cảnh
    scene, confidence = classify_scene_simple(red_image)
    print(f"   - Phân loại phong cảnh: {scene} (confidence: {confidence:.2f})")
    
    # Test chuyển đổi RGB sang tên màu
    test_colors = [
        [255, 0, 0],    # Đỏ
        [0, 255, 0],    # Xanh lá
        [0, 0, 255],    # Xanh dương
        [255, 255, 255], # Trắng
        [0, 0, 0],      # Đen
    ]
    
    for color in test_colors:
        color_name = rgb_to_color_name(np.array(color))
        print(f"   - RGB{color} -> {color_name}")

def test_with_real_image():
    """
    Test với ảnh thật (nếu có)
    """
    # Tìm ảnh trong thư mục data
    data_dir = "data"
    if os.path.exists(data_dir):
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            test_image = os.path.join(data_dir, image_files[0])
            print(f"\n5. Test với ảnh thật: {test_image}")
            
            try:
                recognizer = AIObjectRecognizer()
                result = recognizer.process_image(test_image)
                recognizer.display_results(result)
                print("✓ Test với ảnh thật thành công!")
                return True
            except Exception as e:
                print(f"❌ Lỗi khi test với ảnh thật: {e}")
                return False
        else:
            print("\n5. Không tìm thấy ảnh thật để test")
            return True
    else:
        print("\n5. Thư mục data không tồn tại")
        return True

def main():
    """
    Hàm main cho test
    """
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # Chạy test cơ bản
    basic_test_passed = test_basic_functionality()
    
    # Chạy test với ảnh thật
    real_image_test_passed = test_with_real_image()
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT TEST")
    print("="*60)
    print(f"Test cơ bản: {'✓ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Test ảnh thật: {'✓ PASSED' if real_image_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and real_image_test_passed:
        print("\n🎉 TẤT CẢ TEST ĐỀU THÀNH CÔNG!")
        print("Ứng dụng sẵn sàng sử dụng!")
        return 0
    else:
        print("\n❌ MỘT SỐ TEST THẤT BẠI!")
        print("Vui lòng kiểm tra lại cài đặt và dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
