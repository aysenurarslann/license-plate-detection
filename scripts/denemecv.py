import os
import shutil
import random
from sklearn.model_selection import KFold
from ultralytics import YOLO
import numpy as np

# Veri seti dizini
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Desteklenen görüntülerin uzantıları
image_extensions = ['.jpg', '.jpeg', '.png']

# Tüm görüntü dosyalarını listele
def list_all_images(directory, extensions):
    all_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.join(subdir, file))
    return all_files

images = list_all_images(images_dir, image_extensions)
labels = [os.path.splitext(f)[0] + '.txt' for f in images]

if not images or not labels:
    raise ValueError("No image or label files found in the specified directories.")

data = list(zip(images, labels))
random.shuffle(data)

# K-Fold Çapraz Doğrulama
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(data), 1):
    print(f"\nFold {fold}/{k}")

    # Eğitim ve doğrulama klasörlerini oluştur
    train_images_dir = os.path.join(images_dir, f'train_fold{fold}')
    val_images_dir = os.path.join(images_dir, f'val_fold{fold}')
    train_labels_dir = os.path.join(labels_dir, f'train_fold{fold}')
    val_labels_dir = os.path.join(labels_dir, f'val_fold{fold}')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Dosyaları kopyala
    for idx in train_index:
        image, label = data[idx]
        image_dst = os.path.join(train_images_dir, os.path.basename(image))
        label_dst = os.path.join(train_labels_dir, os.path.basename(label))

        if not os.path.samefile(image, image_dst):
            shutil.copy(image, image_dst)
        if not os.path.samefile(os.path.join(labels_dir, label), label_dst):
            shutil.copy(os.path.join(labels_dir, label), label_dst)

    for idx in val_index:
        image, label = data[idx]
        image_dst = os.path.join(val_images_dir, os.path.basename(image))
        label_dst = os.path.join(val_labels_dir, os.path.basename(label))

        if not os.path.samefile(image, image_dst):
            shutil.copy(image, image_dst)
        if not os.path.samefile(os.path.join(labels_dir, label), label_dst):
            shutil.copy(os.path.join(labels_dir, label), label_dst)

    # YOLO modelini yükle ve eğit
    model = YOLO("best.pt")
    model.train(data={'train': train_images_dir, 'val': val_images_dir},
                epochs=50, batch=16, imgsz=640, project='cross_validation', name=f'fold{fold}')

    # Klasörleri temizle
    shutil.rmtree(train_images_dir)
    shutil.rmtree(val_images_dir)
    shutil.rmtree(train_labels_dir)
    shutil.rmtree(val_labels_dir)

print('Çapraz doğrulama tamamlandı!')