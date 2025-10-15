#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy ví dụ với ảnh thật hoặc ảnh có người thật
"""

import os
import sys
import cv2
import numpy as np
from main import AIObjectRecognizer

def download_sample_image():
    """
    Tải ảnh mẫu từ internet hoặc tạo ảnh có người thật
    """
    try:
        import requests
        from PIL import Image
        import io
        
        # URL ảnh mẫu có người
        sample_urls = [
            "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=800&h=600&fit=crop"
        ]
        
        for i, url in enumerate(sample_urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    image = image.resize((800, 600))
                    output_path = f"data/sample_image_{i+1}.jpg"
                    image.save(output_path)
                    print(f"Da tai anh mau {i+1}: {output_path}")
                    return output_path
            except Exception as e:
                print(f"Loi khi tai anh {i+1}: {e}")
                continue
        
        print("Khong the tai anh mau tu internet")
        return None
        
    except ImportError:
        print("Thieu thu vien requests, dang tao anh mau don gian...")
        return create_simple_person_image()

def create_simple_person_image():
    """
    Tạo ảnh đơn giản có người (sử dụng OpenCV)
    """
    # Tạo ảnh nền
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Vẽ nền trời
    image[0:300, :] = [135, 206, 235]
    
    # Vẽ nền đất
    image[300:, :] = [34, 139, 34]
    
    # Vẽ người đơn giản nhưng có cấu trúc rõ ràng
    # Thân người (hình chữ nhật đen)
    cv2.rectangle(image, (300, 200), (400, 500), (0, 0, 0), -1)
    
    # Đầu (hình tròn)
    cv2.circle(image, (350, 150), 40, (255, 220, 177), -1)
    
    # Áo (hình chữ nhật trắng)
    cv2.rectangle(image, (320, 250), (380, 350), (255, 255, 255), -1)
    
    # Quần (hình chữ nhật xanh)
    cv2.rectangle(image, (320, 350), (380, 500), (0, 0, 255), -1)
    
    # Tay
    cv2.rectangle(image, (280, 250), (300, 350), (255, 220, 177), -1)
    cv2.rectangle(image, (400, 250), (420, 350), (255, 220, 177), -1)
    
    # Chân
    cv2.rectangle(image, (320, 500), (340, 550), (255, 220, 177), -1)
    cv2.rectangle(image, (360, 500), (380, 550), (255, 220, 177), -1)
    
    # Vẽ điện thoại
    cv2.rectangle(image, (420, 300), (440, 380), (50, 50, 50), -1)
    cv2.rectangle(image, (422, 302), (438, 378), (200, 200, 200), -1)
    
    # Vẽ túi xách
    cv2.rectangle(image, (280, 400), (320, 450), (139, 69, 19), -1)
    
    # Lưu ảnh
    output_path = "data/simple_person_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Da tao anh nguoi don gian: {output_path}")
    
    return output_path

def test_with_sample_image():
    """
    Test với ảnh mẫu
    """
    print("="*60)
    print("TEST VOI ANH MAU")
    print("="*60)
    
    # Tạo thư mục data
    os.makedirs("data", exist_ok=True)
    
    # Tải hoặc tạo ảnh mẫu
    sample_image = download_sample_image()
    if sample_image is None:
        sample_image = create_simple_person_image()
    
    try:
        # Khởi tạo recognizer
        print(f"\nDang xu ly anh: {sample_image}")
        recognizer = AIObjectRecognizer()
        
        # Xử lý ảnh
        result = recognizer.process_image(sample_image)
        
        # Hiển thị kết quả
        recognizer.display_results(result)
        
        print("\n" + "="*60)
        print("TEST HOAN THANH!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Loi khi test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Hàm main
    """
    success = test_with_sample_image()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
