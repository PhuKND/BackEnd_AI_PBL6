# Hệ thống tìm kiếm sản phẩm y tế bằng AI

Dự án này tích hợp AI để nhận diện và tìm kiếm sản phẩm y tế thông qua camera hoặc tải ảnh lên.

## Tính năng

- 🔍 Tìm kiếm sản phẩm bằng từ khóa
- 📷 Chụp ảnh trực tiếp từ webcam để nhận diện sản phẩm
- 📁 Tải ảnh từ máy tính để nhận diện sản phẩm
- 🤖 AI phân loại ảnh sử dụng model ResNet đã train
- 📱 Giao diện responsive, thân thiện với mobile

## Cài đặt và chạy

### 1. Cài đặt Backend API

```bash
# Cài đặt dependencies cho API
pip install -r api_requirements.txt

# Chạy API server
python api_server.py
```

API sẽ chạy tại `http://localhost:5000`

### 2. Cài đặt Frontend

```bash
cd Front_End_PBL6-main

# Cài đặt dependencies
npm install

# Chạy development server
npm run dev
```

Frontend sẽ chạy tại `http://localhost:5173`

## Cấu trúc dự án

```
├── api_server.py              # Flask API server
├── api_requirements.txt       # Dependencies cho API
├── NhanDienVatTuYTe_best.pth  # Model AI đã train
├── labels.json                # Danh sách classes
├── norm.json                  # Normalization parameters
├── Front_End_PBL6-main/       # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx     # Header với thanh tìm kiếm
│   │   │   ├── CameraModal.jsx # Modal chụp ảnh
│   │   │   ├── ImageUpload.jsx # Modal tải ảnh
│   │   │   └── SearchResults.jsx # Hiển thị kết quả
│   │   └── App.jsx            # App chính
│   └── package.json
└── dataset/                   # Dataset training
```

## Sử dụng

1. **Tìm kiếm bằng từ khóa**: Nhập từ khóa vào thanh tìm kiếm và nhấn Enter
2. **Chụp ảnh**: Click vào icon camera, cho phép truy cập webcam, chụp ảnh và xử lý
3. **Tải ảnh**: Click vào icon hình ảnh, chọn file ảnh từ máy tính và xử lý
4. **Xem kết quả**: Sau khi AI phân loại, trang sẽ hiển thị kết quả tìm kiếm

## API Endpoints

- `POST /api/classify-image`: Phân loại ảnh và trả về kết quả dự đoán
- `GET /api/health`: Kiểm tra trạng thái API và model

## Model AI

- **Architecture**: ResNet18/ResNet34
- **Classes**: 14 loại trang thiết bị y tế
- **Input**: Ảnh RGB 224x224
- **Output**: Tên sản phẩm và độ tin cậy

## Lưu ý

- Đảm bảo có quyền truy cập camera khi sử dụng tính năng chụp ảnh
- Model AI cần GPU để chạy nhanh hơn (tự động fallback về CPU nếu không có GPU)
- Hỗ trợ các định dạng ảnh: JPG, PNG, GIF
