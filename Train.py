import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

# ================== تنظیمات ==================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = 'images'               # مسیر پوشه‌ی اصلی داده‌ها
MODEL_SAVE_PATH = 'cancer_classifier.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ================== آماده‌سازی داده‌ها ==================
# تبدیل‌های آموزشی (با افزایش داده) و اعتبارسنجی
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# بارگذاری داده‌ها با ImageFolder (فرض بر وجود زیرپوشه‌های کلاس)
dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)

# اگر داده‌ها به train/validation تقسیم نشده‌اند، 80% برای آموزش و 20% برای اعتبارسنجی
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# توجه: برای مجموعه اعتبارسنجی باید از تبدیل مخصوص استفاده کنیم
# چون random_split فقط ایندکس‌ها را جدا می‌کند، باید transform را جداگانه تنظیم کنیم
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"تعداد تصاویر آموزش: {len(train_dataset)}")
print(f"تعداد تصاویر اعتبارسنجی: {len(val_dataset)}")
print(f"کلاس‌ها: {dataset.classes}")  # ['benign', 'malignant']

# ================== ساخت مدل با انتقال‌یادگیری ==================
def create_model(num_classes=2):
    # بارگذاری ResNet18 با وزن‌های پیش‌آموزش‌دیده
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    # فریز کردن تمام لایه‌های پایه
    for param in model.parameters():
        param.requires_grad = False
    # تعویض لایه‌ی آخر برای دو کلاس
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

model = create_model().to(DEVICE)

# تعریف تابع هزینه و بهینه‌ساز
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # فقط لایه‌های جدید به‌روز شوند

# ================== آموزش مدل ==================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    # مرحله آموزش
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # مرحله اعتبارسنجی
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_val_loss = running_loss / total
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

# ================== ذخیره مدل ==================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"مدل در {MODEL_SAVE_PATH} ذخیره شد.")

# ================== رسم نمودارها ==================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), train_accs, label='Train Accuracy')
plt.plot(range(1, EPOCHS+1), val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

# ================== ارزیابی نهایی روی مجموعه اعتبارسنجی ==================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nگزارش طبقه‌بندی:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(all_labels, all_preds))

# ================== تابع پیش‌بینی برای تصاویر جدید ==================
def predict_image(img_path, model, class_names, device):
    """
    پیش‌بینی کلاس یک تصویر
    """
    image = Image.open(img_path).convert('RGB')
    transform = val_transforms  # از همان تبدیل اعتبارسنجی استفاده می‌کنیم
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = class_names[predicted.item()]
    confidence = confidence.item()
    return class_name, confidence

# مثال استفاده (بعد از آموزش یا بارگذاری مدل)
# class_names = dataset.classes
# img_path = 'images/test.jpg'
# pred_class, conf = predict_image(img_path, model, class_names, DEVICE)
# print(f"تصویر: {pred_class} با اطمینان {conf:.2%}")

# ================== پیش‌بینی دسته‌ای روی تمام تصاویر یک پوشه ==================
def batch_predict(folder_path, model, class_names, device):
    results = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            pred_class, conf = predict_image(img_path, model, class_names, device)
            results.append((filename, pred_class, conf))
            print(f"{filename}: {pred_class} با اطمینان {conf:.2%}")
    return results

# اگر بخواهید روی پوشه‌ی images پیش‌بینی انجام دهید (پس از آموزش یا بارگذاری مدل):
# results = batch_predict('images', model, dataset.classes, DEVICE)

# ذخیره نتایج در فایل CSV
import csv
def save_results(results, filename='predictions.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Predicted Class', 'Confidence'])
        writer.writerows(results)
    print(f"نتایج در {filename} ذخیره شد.")

# مثال:
# save_results(results)
