#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng Flask: AI Object Recognition Web App
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import AIObjectRecognizer

# ===============================
# ⚙️ Cấu hình cơ bản
# ===============================
app = Flask(__name__)
app.secret_key = "aiobjectrecognitionsecret"
CORS(app)

# Giới hạn dung lượng upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Đường dẫn upload & kết quả
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "data", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# ===============================
# 🧠 Hàm tiện ích
# ===============================
def allowed_file(filename):
    """Kiểm tra định dạng file hợp lệ"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def to_serializable(obj):
    """
    🔧 Chuyển đổi dữ liệu không JSON-serializable (numpy, tensor...) sang kiểu hợp lệ
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ===============================
# 🧠 Khởi tạo AI model
# ===============================
recognizer = AIObjectRecognizer()

# ===============================
# 🏠 Trang chủ
# ===============================
@app.route('/')
def index():
    return render_template('index.html')


# ===============================
# 📤 Xử lý upload ảnh
# ===============================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("Không có file nào được chọn.")
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("Vui lòng chọn một file ảnh hợp lệ.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"📸 Đã nhận ảnh: {filepath}")

        try:
            # Gọi model AI xử lý ảnh
            result = recognizer.process_image(filepath)
            print("✅ Xử lý ảnh hoàn tất.")

            # Tạo dữ liệu kết quả
            result_data = {
                "filename": filename,
                "original_image": f"/static/{filename}" if os.path.exists(f"static/{filename}") else None,
                "result_image": result.get("result_image"),
                "analysis": result,
            }

            # Lưu kết quả JSON (ép kiểu an toàn)
            result_json_path = os.path.join(RESULT_FOLDER, f"{os.path.splitext(filename)[0]}.json")
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2, default=to_serializable)

            print(f"💾 Đã lưu kết quả vào: {result_json_path}")

            # Hiển thị kết quả
            return render_template('results.html', result=result_data)

        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh: {e}")
            import traceback
            traceback.print_exc()
            flash(f"Lỗi khi xử lý ảnh: {e}")
            return redirect(url_for('index'))

    else:
        flash("Định dạng file không hợp lệ. Vui lòng chọn JPG, PNG, JPEG, GIF hoặc BMP.")
        return redirect(url_for('index'))


# ===============================
# 📜 Trang lịch sử
# ===============================
@app.route('/history')
def history():
    files = [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.json')]
    results = []
    for f in sorted(files, reverse=True):
        try:
            with open(os.path.join(RESULT_FOLDER, f), encoding="utf-8") as infile:
                data = json.load(infile)
                results.append(data)
        except Exception as e:
            print(f"Lỗi đọc file {f}: {e}")
            continue
    return render_template('history.html', results=results)


# ===============================
# 🌐 API Endpoint cho Flutter/Web khác
# ===============================
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API phân tích ảnh và trả về JSON"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"📥 [API] Đã nhận file: {filename}")

    try:
        result = recognizer.process_image(filepath)
        response_data = {"success": True, "analysis": result}

        # Trả về JSON an toàn
        return Response(
            json.dumps(response_data, ensure_ascii=False, indent=2, default=to_serializable),
            mimetype="application/json"
        )
    except Exception as e:
        print(f"❌ [API] Lỗi xử lý ảnh: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ===============================
# 🚀 Chạy ứng dụng
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
