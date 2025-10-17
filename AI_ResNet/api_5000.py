from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import io
import base64

app = Flask(__name__)
CORS(app)

app.config['JSON_AS_ASCII'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'NhanDienVatTuYTe_best.pth')
LABELS_PATH = os.path.join(BASE, 'labels.json')
NORM_PATH = os.path.join(BASE, 'norm.json')

model = None
classes = None
mean = None
std = None
img_size = None
device = "cuda" if torch.cuda.is_available() else "cpu"

ENG2VI = {
    "blood pressure monitor": "máy đo huyết áp",
    "cotton balls": "bông gòn",
    "infrared thermometer": "nhiệt kế hồng ngoại",
    "medical gloves": "găng tay y tế",
    "medical mask": "khẩu trang",
    "medical tape": "băng keo y tế",
    "medical tweezers": "nhíp y tế",
    "medicine cup": "cốc y tế",
    "mercury thermometer": "nhiệt kế thủy ngân",
    "nebulizer mask": "mặt nạ máy xông",
    "pulse oximeter": "máy đo độ bão hòa oxy",
    "reflex hammer": "búa phản xạ",
    "stethoscope": "ống nghe",
    "surgical scissors": "kéo phẫu thuật",
}

def _norm_label(s: str) -> str:
    """Chuẩn hóa nhãn để tra cứu mapping (lowercase, thay '_' thành ' ', strip)."""
    return str(s).replace("_", " ").strip().lower()

def build_model(arch: str, num_classes: int):
    if arch == 'resnet34':
        m = models.resnet34(weights=None)
    else:
        m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def load_bundle(model_path: str, labels_path: str, norm_path: str, device: str = 'cpu'):
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt.get('classes', None)
    if classes is None and os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            classes = json.load(f)['classes']

    mean = ckpt.get('mean', None)
    std = ckpt.get('std', None)
    if (mean is None or std is None) and os.path.exists(norm_path):
        with open(norm_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            mean = d['mean']
            std = d['std']
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    img_size = int(ckpt.get('img_size', 224))
    arch = str(ckpt.get('arch', 'resnet18'))

    model = build_model(arch, len(classes)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, classes, mean, std, img_size

def make_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

@torch.no_grad()
def predict_tensor(model: torch.nn.Module, x: torch.Tensor, device: str):
    y = model(x.to(device)).softmax(1)[0]
    conf = float(torch.max(y).item())
    idx = int(torch.argmax(y).item())
    return idx, conf

def load_image_from_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Không thể đọc ảnh: {str(e)}")

def initialize_model():
    global model, classes, mean, std, img_size
    try:
        model, classes, mean, std, img_size = load_bundle(MODEL_PATH, LABELS_PATH, NORM_PATH, device)
        print(f"Model loaded successfully on {device}")
        print(f"Classes: {classes}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def en_to_vi(label_en: str) -> str:
    """
    Trả về nhãn tiếng Việt tương ứng.
    - Ưu tiên map theo các nhãn đã liệt kê.
    - Nếu không tìm thấy, trả về chính nhãn đầu vào (fallback).
    """
    return ENG2VI.get(_norm_label(label_en), label_en)

@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400

        image_bytes = file.read()
        pil_image = load_image_from_bytes(image_bytes)

        transform = make_transform(img_size, mean, std)
        x = transform(pil_image).unsqueeze(0)

        idx, conf = predict_tensor(model, x, device)
        prediction_en = classes[idx]
        prediction_vi = en_to_vi(prediction_en)
        confidence = round(conf * 100, 2)

        app.logger.info(
            "AI → %s | %s (%.2f%%) | idx=%d | client=%s | image=%dx%d",
            prediction_vi, prediction_en, confidence, idx, request.remote_addr,
            pil_image.width, pil_image.height
        )

        return jsonify({
            'prediction_vi': prediction_vi,     
            'prediction_en': prediction_en,     
            'confidence': confidence,
            'class_index': idx
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'classes_count': len(classes) if classes else 0
    })

if __name__ == '__main__':
    print("Initializing model...")
    initialize_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
