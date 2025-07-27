import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

def get_class_names_from_folder(dataset_dir):
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()  # Đảm bảo đúng thứ tự index như ImageFolder
    return class_names

# Đường dẫn tới thư mục chứa các folder class
DATASET_DIR = r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Documents\Plant_Disease_Detection\MyAI\PlantVillage"
disease_names = get_class_names_from_folder(DATASET_DIR)
print("Tên class sẽ được dùng:", disease_names)
NUM_CLASSES = len(disease_names)

# Khai báo lại kiến trúc model giống file train!
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out

# Số class, lấy đúng số đã train
NUM_CLASSES = 15  # Nếu số class khác, sửa lại cho đúng
# Load model
model = CNN(NUM_CLASSES)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'plant_disease_detection_model.pt'), map_location='cpu'))
model.eval()

# # Đọc tên class từ file disease_info.csv (nếu có)
# # Nếu không có, dùng list tên class từ dataset train ban đầu (dataset.class_to_idx)
# disease_names = []
# csv_path = os.path.join(os.path.dirname(__file__), 'disease_info.csv')
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path, encoding="cp1252")
#     disease_names = df["disease_name"].tolist()
# else:
#     # Nếu không có file, dùng index số 0...NUM_CLASSES-1
#     disease_names = [f"Class {i}" for i in range(NUM_CLASSES)]

# Transform giống lúc train
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "Không tìm thấy file tải lên!"
            return render_template('index.html', result=result)
        file = request.files['file']
        if file.filename == '':
            result = "Chưa chọn file!"
            return render_template('index.html', result=result)
        if file:
            img = Image.open(file.stream).convert("RGB")
            img_tran = transform(img)
            img_tran = img_tran.unsqueeze(0)
            with torch.no_grad():
                output = model(img_tran)
                pred_idx = output.argmax(dim=1).item()
                result = disease_names[pred_idx]
            # Lưu file vào static để show lại
            filename = os.path.join('static', 'upload.jpg')
            img.save(os.path.join(os.path.dirname(__file__), filename))
            return render_template('index.html', result=result, image_file=filename)
    return render_template('index.html', result=result)

# Cho phép chạy trực tiếp bằng python flaskr (nếu muốn)
if __name__ == '__main__':
    app.run(debug=True)