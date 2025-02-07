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

# Alt dizinlerdeki tüm görüntü dosyalarını listele
def list_all_images(directory, extensions):
    all_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.relpath(os.path.join(subdir, file), directory))
    if not all_files:
        print(f"No image files found in {directory}.")
    return all_files

images = list_all_images(images_dir, image_extensions)
labels = [os.path.splitext(f)[0] + '.txt' for f in images]

# Boş veri kontrolü
if not images:
    raise ValueError("No image files found in the specified directory.")
if not labels:
    raise ValueError("No label files found in the specified directory.")

data = list(zip(images, labels))
random.shuffle(data)

# K-Fold Çapraz Doğrulama
k = 2  # Kat sayısı
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_scores = []  # Her fold için doğruluk değerlerini saklayacak liste

fold = 0
for train_index, val_index in kf.split(data):
    fold += 1
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

    # Eğitim dosyalarını kopyala
    for idx in train_index:
        image, label = data[idx]
        image_src = os.path.join(images_dir, image)
        label_src = os.path.join(labels_dir, label)
        image_dst = os.path.join(train_images_dir, os.path.basename(image))
        label_dst = os.path.join(train_labels_dir, os.path.basename(label))

        if os.path.exists(image_src) and not os.path.samefile(image_src, image_dst):
            shutil.copy(image_src, image_dst)
        else:
            print(f"Uyarı: {image},{images_dir} da bulunamadı veya kaynak ve hedef aynı")

        if os.path.exists(label_src) and not os.path.samefile(label_src, label_dst):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Warning: {label} not found in {labels_dir} or source and destination are the same")

    # Doğrulama dosyalarını kopyala
    for idx in val_index:
        image, label = data[idx]
        image_src = os.path.join(images_dir, image)
        label_src = os.path.join(labels_dir, label)
        image_dst = os.path.join(val_images_dir, os.path.basename(image))
        label_dst = os.path.join(val_labels_dir, os.path.basename(label))

        if os.path.exists(image_src) and not os.path.samefile(image_src, image_dst):
            shutil.copy(image_src, image_dst)
        else:
            print(f"Warning: {image} not found in {images_dir} or source and destination are the same")

        if os.path.exists(label_src) and not os.path.samefile(label_src, label_dst):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Warning: {label} not found in {labels_dir} or source and destination are the same")

    # YOLO modelini yükle ve eğit
    model = YOLO("best.pt")  # Burada daha önce eğitilmiş model dosyasını kullanıyoruz
    model.train(data={
        'train': train_images_dir,
        'val': val_images_dir
    }, epochs=50, batch=16, imgsz=640, project='cross_validation', name=f'fold{fold}')

    # Modelin değerlendirme ve doğruluk skoru hesaplama
    # Örneğin:
    # results = model.val(data=val_images_dir)
    # accuracy = results.metrics.accuracy
    # accuracy_scores.append(accuracy)
    
    # Klasörleri temizle
    shutil.rmtree(train_images_dir)
    shutil.rmtree(val_images_dir)
    shutil.rmtree(train_labels_dir)
    shutil.rmtree(val_labels_dir)

# Ortalama Doğruluk
print(f'Average Accuracy: {np.mean(accuracy_scores)}')





# Kopyalama işlemi, K-Fold çapraz doğrulama sırasında her bir fold (kat) için veri setinin eğitim ve 
# doğrulama alt kümelerine ayrılmasını sağlar. Her fold, modelin farklı eğitim ve doğrulama veri alt 
# kümeleri üzerinde test edilmesini ve sonuçların değerlendirilmesini sağlar. Ancak bu işlem doğrudan 
# orijinal dosyalar üzerinde yapılmaz; bunun yerine, veriler geçici olarak belirli klasörlere kopyalanır. 
# Bu sayede her fold için verilerin doğru şekilde organize edilmesi ve modelin eğitilmesi sağlanır. Ayrıca,
# her fold sonrası bu geçici dosyalar silinir.