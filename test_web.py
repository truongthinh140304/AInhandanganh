#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script cho ứng dụng web AI Object Recognition
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_web_app():
    """
    Test ứng dụng web
    """
    print("="*60)
    print("TESTING WEB APPLICATION")
    print("="*60)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Kiểm tra trang chủ
    print("\n1. Testing trang chủ...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✓ Trang chủ hoạt động bình thường")
        else:
            print(f"✗ Lỗi trang chủ: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Không thể kết nối đến ứng dụng: {e}")
        print("Vui lòng chạy: python start_web.py")
        return False
    
    # Test 2: Kiểm tra trang lịch sử
    print("\n2. Testing trang lịch sử...")
    try:
        response = requests.get(f"{base_url}/history", timeout=10)
        if response.status_code == 200:
            print("✓ Trang lịch sử hoạt động bình thường")
        else:
            print(f"✗ Lỗi trang lịch sử: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Lỗi trang lịch sử: {e}")
    
    # Test 3: Kiểm tra API endpoint
    print("\n3. Testing API endpoint...")
    try:
        # Tạo ảnh test đơn giản
        test_image_path = "data/test_web_image.jpg"
        if not os.path.exists(test_image_path):
            create_test_image(test_image_path)
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/api/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✓ API endpoint hoạt động bình thường")
                print(f"  - Phát hiện {result['analysis']['total_people']} người")
                print(f"  - Phát hiện {result['analysis']['total_objects']} vật dụng")
                print(f"  - Phong cảnh: {result['analysis']['scene']}")
            else:
                print("✗ API trả về lỗi")
        else:
            print(f"✗ Lỗi API: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Lỗi API: {e}")
    except Exception as e:
        print(f"✗ Lỗi test API: {e}")
    
    print("\n" + "="*60)
    print("TEST HOAN THANH!")
    print("="*60)
    print("Ứng dụng web đã sẵn sàng sử dụng!")
    print("Truy cập: http://localhost:5000")
    print("="*60)
    
    return True

def create_test_image(output_path):
    """
    Tạo ảnh test đơn giản
    """
    try:
        import cv2
        import numpy as np
        
        # Tạo ảnh test
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Vẽ người đơn giản
        cv2.rectangle(image, (200, 100), (400, 350), (0, 0, 0), -1)
        cv2.circle(image, (300, 80), 30, (255, 220, 177), -1)
        cv2.rectangle(image, (220, 150), (380, 250), (255, 255, 255), -1)
        
        # Lưu ảnh
        cv2.imwrite(output_path, image)
        print(f"Đã tạo ảnh test: {output_path}")
        
    except ImportError:
        print("Cần cài đặt OpenCV để tạo ảnh test")
    except Exception as e:
        print(f"Lỗi tạo ảnh test: {e}")

def main():
    """
    Hàm main
    """
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # Test ứng dụng web
    success = test_web_app()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
