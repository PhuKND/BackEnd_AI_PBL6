import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True, 
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float32,
)
model.to(device)
model.eval()

label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.json")
with open(label_encoder_path, "r", encoding="utf-8") as f:
    label_data = json.load(f)

idx2label = {i: label for i, label in enumerate(label_data["classes"])}

def predict(text: str, max_len: int = 256):
    """
    Input: câu mô tả triệu chứng (tiếng Việt tự nhiên).
    Output:
      - Nhãn bệnh dự đoán
      - Độ tự tin (softmax probability)
      - Phân bố xác suất cho tất cả nhãn
    """

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)[0] 

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx2label[pred_idx]
    pred_confidence = float(probs[pred_idx].item())

    all_probs = [
        {
            "label": idx2label[i],
            "p": float(probs[i].item()),
        }
        for i in range(len(idx2label))
    ]

    return {
        "input_text": text,
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "pred_confidence": pred_confidence,
        "all_probs": all_probs,
    }

if __name__ == "__main__":
    print("=== HỆ THỐNG GỢI Ý BỆNH DỰA TRÊN TRIỆU CHỨNG (PhoBERT) ===")
    print("Nhập mô tả triệu chứng bằng tiếng Việt tự nhiên.")
    print("Ví dụ: 'Em bị sốt 38 độ từ hôm qua, đau họng và ho khan liên tục.'")
    print("Gõ trống (Enter) để thoát.\n")

    THRESHOLD = 0.5 

    while True:
        user_text = input("Triệu chứng của bệnh nhân: ").strip()

        if user_text == "":
            print("Tạm biệt.")
            break

        result = predict(user_text)

        print("\n=== KẾT QUẢ DỰ ĐOÁN ===")

        if result['pred_confidence'] < THRESHOLD:
            print("→ Triệu chứng của loại bệnh này lạ quá, có lẽ tôi chưa được huấn luyện về triệu chứng này ạ.")
            print(f"(Độ tự tin hiện tại: {result['pred_confidence']:.4f})")
        else:
            print(f"→ Bệnh dự đoán: {result['pred_label']} (class {result['pred_idx']})")
            print(f"→ Độ tự tin: {result['pred_confidence']:.4f}")

        print("Phân bố xác suất:")
        for row in result["all_probs"]:
            print(f"  {row['label']}: {row['p']:.4f}")
        print()
