# 🎓 Hướng dẫn Training Model

Hướng dẫn chi tiết để nâng cấp và training các model AI cho ứng dụng nhận dạng đối tượng.

## 📋 Mục lục

1. [Training Model Phân loại Giới tính](#training-model-phân-loại-giới-tính)
2. [Training Model Phân loại Phong cảnh](#training-model-phân-loại-phong-cảnh)
3. [Fine-tuning YOLO](#fine-tuning-yolo)
4. [Dataset và Dữ liệu](#dataset-và-dữ-liệu)
5. [Evaluation và Testing](#evaluation-và-testing)

## 🧑‍🤝‍🧑 Training Model Phân loại Giới tính

### 1. Chuẩn bị Dataset

#### Dataset đề xuất:
- **UTKFace**: 20,000+ ảnh khuôn mặt với label giới tính và tuổi
- **CelebA**: 200,000+ ảnh celebrity với 40 attributes
- **WIDER Face**: Dataset khuôn mặt đa dạng

#### Cấu trúc thư mục:
```
gender_dataset/
├── train/
│   ├── male/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── female/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── val/
│   ├── male/
│   └── female/
└── test/
    ├── male/
    └── female/
```

### 2. Code Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

class GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        # Sử dụng ResNet18 pretrained
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def train_gender_model():
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_dataset = ImageFolder('gender_dataset/train', transform=transform)
    val_dataset = ImageFolder('gender_dataset/val', transform=transform)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = GenderClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_acc = 0
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
        
        # Print results
        train_acc = 100. * train_correct / len(train_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/gender_classifier_best.pth')
        
        scheduler.step()
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_gender_model()
```

## 🏞️ Training Model Phân loại Phong cảnh

### 1. Dataset đề xuất:
- **Places365**: 365 categories với 1.8M ảnh
- **MIT Indoor Scene**: 67 categories indoor scenes
- **Weather Dataset**: Nắng, mưa, u ám, tuyết

### 2. Code Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class SceneClassifier(nn.Module):
    def __init__(self, num_classes=6):  # indoor, outdoor_sunny, outdoor_cloudy, outdoor_rainy, outdoor_snowy, outdoor_night
        super(SceneClassifier, self).__init__()
        # Sử dụng EfficientNet
        self.backbone = torchvision.models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def train_scene_model():
    # Transform với augmentation mạnh hơn
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_dataset = ImageFolder('scene_dataset/train', transform=transform)
    val_dataset = ImageFolder('scene_dataset/val', transform=transform)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model
    model = SceneClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_acc = 0
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
        
        # Print results
        train_acc = 100. * train_correct / len(train_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/scene_classifier_best.pth')
        
        scheduler.step()
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_scene_model()
```

## 🎯 Fine-tuning YOLO

### 1. Chuẩn bị Dataset YOLO

#### Format annotation (YOLO):
```
# class_id center_x center_y width height (normalized)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

#### Cấu trúc thư mục:
```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

### 2. Code Fine-tuning

```python
from ultralytics import YOLO
import yaml

def fine_tune_yolo():
    # Tạo file config
    config = {
        'path': 'yolo_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,  # number of classes
        'names': ['person', 'object']
    }
    
    with open('yolo_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Fine-tune
    results = model.train(
        data='yolo_config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        cache=False,
        device='',
        workers=8,
        project='runs/train',
        name='yolo_finetuned',
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        cache=False,
        device='',
        workers=8,
        project='runs/train',
        name='yolo_finetuned',
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False
    )
    
    print("Fine-tuning completed!")
    return results

if __name__ == "__main__":
    fine_tune_yolo()
```

## 📊 Dataset và Dữ liệu

### 1. Thu thập dữ liệu

#### Nguồn dữ liệu:
- **Kaggle**: Competitions và datasets
- **Google Images**: Tìm kiếm và tải về
- **Unsplash**: Ảnh miễn phí chất lượng cao
- **Pexels**: Ảnh stock miễn phí
- **Flickr**: API để tải ảnh

#### Script thu thập dữ liệu:
```python
import requests
from PIL import Image
import io
import os

def download_images(query, num_images=100, output_dir="dataset"):
    """
    Tải ảnh từ Unsplash API
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unsplash API (cần đăng ký để có access key)
    url = "https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": "Client-ID YOUR_ACCESS_KEY"
    }
    params = {
        "query": query,
        "per_page": num_images,
        "orientation": "landscape"
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    for i, photo in enumerate(data['results']):
        try:
            img_url = photo['urls']['regular']
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img.save(f"{output_dir}/{query}_{i:03d}.jpg")
            print(f"Downloaded {query}_{i:03d}.jpg")
        except Exception as e:
            print(f"Error downloading image {i}: {e}")

# Sử dụng
download_images("person outdoor", 50, "dataset/person_outdoor")
download_images("person indoor", 50, "dataset/person_indoor")
```

### 2. Annotation và Labeling

#### Công cụ annotation:
- **LabelImg**: GUI tool cho YOLO format
- **CVAT**: Web-based annotation tool
- **Roboflow**: Platform annotation và augmentation
- **Supervisely**: Professional annotation platform

#### Script tự động annotation:
```python
import cv2
import json
import os

def auto_annotate_person(image_path, output_path):
    """
    Tự động annotation người bằng OpenCV
    """
    # Load ảnh
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    annotations = []
    for (x, y, w, h) in faces:
        # Chuyển đổi sang YOLO format
        img_h, img_w = image.shape[:2]
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h
        
        annotations.append({
            'class_id': 0,  # person
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        })
    
    # Lưu annotation
    with open(output_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {ann['center_x']} {ann['center_y']} {ann['width']} {ann['height']}\n")
    
    return len(annotations)

# Sử dụng
auto_annotate_person("image.jpg", "image.txt")
```

## 📈 Evaluation và Testing

### 1. Metrics đánh giá

#### Classification:
- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Độ chính xác cho từng class
- **Recall**: Độ nhạy cho từng class
- **F1-Score**: Harmonic mean của precision và recall
- **Confusion Matrix**: Ma trận nhầm lẫn

#### Object Detection:
- **mAP (mean Average Precision)**: Độ chính xác trung bình
- **IoU (Intersection over Union)**: Độ chồng lấp
- **Precision-Recall Curve**: Đường cong precision-recall

### 2. Code Evaluation

```python
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_model(model, test_loader, device):
    """
    Đánh giá model phân loại
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Tính metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def evaluate_yolo_model(model, test_images, conf_threshold=0.5):
    """
    Đánh giá model YOLO
    """
    results = model(test_images, conf=conf_threshold)
    
    # Tính mAP
    mAP = results[0].box.map if hasattr(results[0].box, 'map') else 0
    
    # Tính precision và recall
    precision = results[0].box.mp if hasattr(results[0].box, 'mp') else 0
    recall = results[0].box.mr if hasattr(results[0].box, 'mr') else 0
    
    print(f"mAP: {mAP:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall
    }

# Sử dụng
# evaluate_classification_model(gender_model, test_loader, device)
# evaluate_yolo_model(yolo_model, test_images)
```

## 🚀 Deployment và Production

### 1. Optimize Model

#### Quantization:
```python
import torch.quantization as quantization

def quantize_model(model):
    """
    Quantize model để giảm kích thước và tăng tốc độ
    """
    model.eval()
    
    # Dynamic quantization
    quantized_model = quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    return quantized_model

# Sử dụng
# quantized_gender_model = quantize_model(gender_model)
```

#### ONNX Export:
```python
import torch.onnx

def export_to_onnx(model, input_shape, output_path):
    """
    Export model sang ONNX format
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

# Sử dụng
# export_to_onnx(gender_model, (3, 224, 224), "models/gender_model.onnx")
```

### 2. API Server

#### FastAPI Implementation:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(title="AI Object Recognition API")

# Load models
gender_model = torch.load('models/gender_classifier_best.pth')
scene_model = torch.load('models/scene_classifier_best.pth')
yolo_model = YOLO('models/yolo_finetuned.pt')

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    API endpoint để phân tích ảnh
    """
    try:
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Phân tích
        result = process_image(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def process_image(image):
    """
    Xử lý ảnh và trả về kết quả
    """
    # YOLO detection
    yolo_results = yolo_model(image)
    
    # Gender classification
    gender_results = classify_gender(image, yolo_results)
    
    # Scene classification
    scene_result = classify_scene(image)
    
    return {
        "detections": yolo_results,
        "gender_results": gender_results,
        "scene": scene_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📚 Tài liệu tham khảo

### Papers:
1. **YOLOv8**: "YOLOv8: A New State-of-the-Art Real-Time Object Detection Model"
2. **ResNet**: "Deep Residual Learning for Image Recognition"
3. **EfficientNet**: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

### Datasets:
1. **UTKFace**: https://susanqq.github.io/UTKFace/
2. **CelebA**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
3. **Places365**: http://places2.csail.mit.edu/
4. **COCO**: https://cocodataset.org/

### Tools:
1. **LabelImg**: https://github.com/tzutalin/labelImg
2. **CVAT**: https://github.com/openvinotoolkit/cvat
3. **Roboflow**: https://roboflow.com/
4. **Weights & Biases**: https://wandb.ai/

---

**Chúc bạn training thành công! 🎉**
