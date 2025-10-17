import os, json, torch, numpy as np, cv2
from PIL import Image
from torchvision import transforms, models

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'NhanDienVatTuYTe_best.pth')
LABELS_PATH = os.path.join(BASE, 'labels.json')
NORM_PATH = os.path.join(BASE, 'norm.json')

IMAGE_PATH = os.path.join(BASE, '213123.png') 
SAVE_OUT_PATH = os.path.join(BASE, 'Screenshot 2025-09-19 135439.png')  

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

    mean = ckpt.get('mean', None); std = ckpt.get('std', None)
    if (mean is None or std is None) and os.path.exists(norm_path):
        with open(norm_path, 'r', encoding='utf-8') as f:
            d = json.load(f); mean = d['mean']; std = d['std']
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

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

def draw_text_bgr(frame_bgr: np.ndarray, text: str, org=(10, 30)) -> np.ndarray:
    out = frame_bgr.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def load_image_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    im = Image.open(path).convert('RGB')
    return im

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, classes, mean, std, img_size = load_bundle(MODEL_PATH, LABELS_PATH, NORM_PATH, device)
    tfm = make_transform(img_size, mean, std)

    pil = load_image_rgb(IMAGE_PATH)
    x = tfm(pil).unsqueeze(0)
    idx, conf = predict_tensor(model, x, device)
    label = f"{classes[idx]}  {conf*100:.1f}%"

    print("Prediction:", label)

    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    out = draw_text_bgr(bgr, label, (10, 30))
    cv2.imwrite(SAVE_OUT_PATH, out)
    print(f"Saved annotated image to: {SAVE_OUT_PATH}")

if __name__ == "__main__":
    main()
