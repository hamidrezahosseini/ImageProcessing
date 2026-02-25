import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18  # یا مدلی که استفاده کردید
from PIL import Image
import csv

# ================== تنظیمات ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
MODEL_PATH = 'cancer_classifier.pth'   # مسیر فایل مدل آموزش‌دیده
INPUT_FOLDER = 'new_images'             # پوشه حاوی تصاویر جدید
OUTPUT_CSV = 'predictions.csv'          # فایل خروجی
CLASS_NAMES = ['benign', 'malignant']   # نام کلاس‌ها (به همان ترتیب training)

# ================== تعریف معماری مدل (باید دقیقاً مشابه مدل آموزش‌دیده باشد) ==================
def create_model(num_classes=2):
    # اگر از ResNet18 استفاده کرده‌اید:
    model = resnet18(weights=None)  # بدون پیش‌آموزش، چون وزن‌ها را خودمان بار می‌کنیم
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

# اگر از MobileNetV2 استفاده کرده‌اید، تابع زیر را جایگزین کنید:
# def create_model(num_classes=2):
#     from torchvision.models import mobilenet_v2
#     model = mobilenet_v2(weights=None)
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.6),
#         nn.Linear(model.last_channel, 128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(128, num_classes)
#     )
#     return model

# ================== بارگذاری مدل ==================
model = create_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"مدل از {MODEL_PATH} بارگذاری شد.")

# ================== تبدیل تصاویر (مطابق validation transforms) ==================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================== تابع پیش‌بینی یک تصویر ==================
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # اضافه کردن بعد batch
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return CLASS_NAMES[predicted.item()], confidence.item()

# ================== پردازش تمام تصاویر پوشه ==================
results = []
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(valid_extensions):
        img_path = os.path.join(INPUT_FOLDER, filename)
        pred_class, conf = predict_image(img_path)
        results.append([filename, pred_class, f"{conf:.2%}"])
        print(f"{filename}: {pred_class} با اطمینان {conf:.2%}")

# ================== ذخیره نتایج در CSV ==================
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Predicted Class', 'Confidence'])
    writer.writerows(results)

print(f"\nنتایج در {OUTPUT_CSV} ذخیره شد.")