#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động nhanh cho ứng dụng web AI Object Recognition
"""

import os
import sys
import webbrowser
import time
import subprocess
from threading import Timer

def open_browser():
    """Mở trình duyệt sau 3 giây"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

def main():
    """Hàm main"""
    print("="*60)
    print("AI OBJECT RECOGNITION WEB APPLICATION")
    print("="*60)
    print("Đang khởi động ứng dụng web...")
    print("Ứng dụng sẽ mở tại: http://localhost:5000")
    print("="*60)
    
    # Tạo thư mục cần thiết
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Mở trình duyệt sau 3 giây
    Timer(3.0, open_browser).start()
    
    # Chạy ứng dụng Flask
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Ứng dụng đã được dừng.")
        print("="*60)
    except Exception as e:
        print(f"Lỗi khi khởi động ứng dụng: {e}")
        print("Vui lòng kiểm tra lại cài đặt.")

if __name__ == "__main__":
    main()
