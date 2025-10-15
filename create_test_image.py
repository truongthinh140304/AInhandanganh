#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo ảnh test phức tạp hơn cho AI Object Recognition
"""

import cv2
import numpy as np
import os

def create_complex_test_image():
    """
    Tạo ảnh test phức tạp hơn với nhiều đối tượng
    """
    # Tạo ảnh nền xanh lá (ngoài trời)
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Nền trời xanh
    image[0:300, :] = [135, 206, 235]  # Sky blue
    
    # Nền đất xanh lá
    image[300:, :] = [34, 139, 34]  # Forest green
    
    # Vẽ mặt trời
    cv2.circle(image, (700, 100), 50, (255, 255, 0), -1)
    
    # Vẽ người 1 (nam) - hình chữ nhật đen với áo trắng
    # Thân người
    cv2.rectangle(image, (200, 200), (280, 450), (0, 0, 0), -1)
    # Áo trắng
    cv2.rectangle(image, (220, 250), (260, 350), (255, 255, 255), -1)
    # Đầu
    cv2.circle(image, (240, 180), 30, (255, 220, 177), -1)
    # Mắt
    cv2.circle(image, (230, 170), 3, (0, 0, 0), -1)
    cv2.circle(image, (250, 170), 3, (0, 0, 0), -1)
    # Miệng
    cv2.ellipse(image, (240, 190), (10, 5), 0, 0, 180, (0, 0, 0), 2)
    
    # Vẽ người 2 (nữ) - hình chữ nhật đen với áo hồng
    # Thân người
    cv2.rectangle(image, (400, 200), (480, 450), (0, 0, 0), -1)
    # Áo hồng
    cv2.rectangle(image, (420, 250), (460, 350), (255, 192, 203), -1)
    # Đầu
    cv2.circle(image, (440, 180), 30, (255, 220, 177), -1)
    # Mắt
    cv2.circle(image, (430, 170), 3, (0, 0, 0), -1)
    cv2.circle(image, (450, 170), 3, (0, 0, 0), -1)
    # Miệng
    cv2.ellipse(image, (440, 190), (10, 5), 0, 0, 180, (0, 0, 0), 2)
    
    # Vẽ điện thoại gần người 1
    cv2.rectangle(image, (300, 300), (320, 360), (50, 50, 50), -1)
    cv2.rectangle(image, (302, 302), (318, 358), (200, 200, 200), -1)
    
    # Vẽ túi xách gần người 2
    cv2.rectangle(image, (500, 350), (550, 400), (139, 69, 19), -1)
    cv2.rectangle(image, (520, 360), (530, 390), (160, 82, 45), -1)
    
    # Vẽ cây
    cv2.rectangle(image, (100, 300), (120, 450), (139, 69, 19), -1)
    cv2.circle(image, (110, 280), 40, (34, 139, 34), -1)
    
    # Vẽ nhà
    cv2.rectangle(image, (50, 350), (150, 450), (160, 82, 45), -1)
    cv2.rectangle(image, (50, 350), (150, 300), (139, 0, 0), -1)  # Mái nhà
    
    # Lưu ảnh
    output_path = "data/complex_test_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Da tao anh test phuc tap tai: {output_path}")
    
    return output_path

def create_indoor_test_image():
    """
    Tạo ảnh test trong nhà
    """
    # Tạo ảnh nền xám (trong nhà)
    image = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # Vẽ tường
    cv2.rectangle(image, (0, 0), (800, 400), (220, 220, 220), -1)
    
    # Vẽ sàn
    cv2.rectangle(image, (0, 400), (800, 600), (139, 69, 19), -1)
    
    # Vẽ cửa sổ
    cv2.rectangle(image, (100, 100), (200, 200), (135, 206, 235), -1)
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 0), 2)
    
    # Vẽ người trong nhà
    cv2.rectangle(image, (400, 300), (480, 500), (0, 0, 0), -1)
    cv2.rectangle(image, (420, 350), (460, 420), (255, 255, 255), -1)  # Áo trắng
    cv2.circle(image, (440, 270), 30, (255, 220, 177), -1)
    
    # Vẽ bàn
    cv2.rectangle(image, (300, 450), (500, 460), (139, 69, 19), -1)
    cv2.rectangle(image, (300, 460), (500, 500), (160, 82, 45), -1)
    
    # Vẽ ghế
    cv2.rectangle(image, (350, 400), (450, 450), (139, 69, 19), -1)
    
    # Vẽ laptop trên bàn
    cv2.rectangle(image, (350, 420), (450, 440), (50, 50, 50), -1)
    
    # Lưu ảnh
    output_path = "data/indoor_test_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Da tao anh test trong nha tai: {output_path}")
    
    return output_path

def main():
    """
    Tạo các ảnh test
    """
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    print("Tao cac anh test...")
    
    # Tạo ảnh test phức tạp
    complex_image = create_complex_test_image()
    
    # Tạo ảnh test trong nhà
    indoor_image = create_indoor_test_image()
    
    print("\nDa tao thanh cong cac anh test:")
    print(f"- Anh ngoai troi: {complex_image}")
    print(f"- Anh trong nha: {indoor_image}")
    
    print("\nBan co the test ung dung voi:")
    print(f"python main.py --image {complex_image}")
    print(f"python main.py --image {indoor_image}")

if __name__ == "__main__":
    main()
