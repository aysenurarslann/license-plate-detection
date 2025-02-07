import os
import shutil
import random
from sklearn.model_selection import KFold

# K-Fold ayarları
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Veri seti dizini
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Train ve validation dizinleri
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')

# Train ve validation label dizinleri
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Desteklenen görüntü uzantıları
image_extensions = ['.jpg', '.jpeg', '.png']
train_images = [f for f in os.listdir(train_images_dir) if any(f.endswith(ext) for ext in image_extensions)]
val_images = [f for f in os.listdir(val_images_dir) if any(f.endswith(ext) for ext in image_extensions)]

# Görüntülerin sayısını kontrol et
print(f"Toplam eğitim görüntü sayısı: {len(train_images)}")
print(f"Toplam doğrulama görüntü sayısı: {len(val_images)}")

if len(train_images) == 0 or len(val_images) == 0:
    raise ValueError("Eğitim veya doğrulama görüntü dosyaları bulunamadı. Lütfen 'train' ve 'val' dizinlerinde görüntü dosyalarının olduğundan emin olun.")

# Eğitim ve doğrulama dosyalarını birleştir
images = train_images + val_images
random.shuffle(images)

# K-Fold bölme işlemi
for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    # Fold için klasörleri oluştur
    fold_train_images_dir = os.path.join(images_dir, f'train_fold_{fold}')
    fold_val_images_dir = os.path.join(images_dir, f'val_fold_{fold}')
    fold_train_labels_dir = os.path.join(labels_dir, f'train_fold_{fold}')
    fold_val_labels_dir = os.path.join(labels_dir, f'val_fold_{fold}')
    
    os.makedirs(fold_train_images_dir, exist_ok=True)
    os.makedirs(fold_val_images_dir, exist_ok=True)
    os.makedirs(fold_train_labels_dir, exist_ok=True)
    os.makedirs(fold_val_labels_dir, exist_ok=True)

    # Eğitim ve doğrulama dosyalarını taşı
    # K-Fold çapraz doğrulama sürecinde her fold için eğitim veri setini hazırlamak amacıyla
    # eğitim ve doğrulama dizinlerinden dosyaları uygun fold dizinlerine kopyalar. 
    # Bu işlem, her fold için ayrı eğitim ve doğrulama veri setleri oluşturmayı sağlar.
    for idx in train_idx:
        image = images[idx]
        label = os.path.splitext(image)[0] + '.txt'
        if os.path.exists(os.path.join(train_images_dir, image)): #Görüntü dosyasının eğitim dizininde olup 
            shutil.copy(os.path.join(train_images_dir, image), fold_train_images_dir) #olmadığını kontrol etmek ve varsa fold eğitim 
                                                                                      # dizinine kopyalamak.
            if os.path.exists(os.path.join(train_labels_dir, label)):
                shutil.copy(os.path.join(train_labels_dir, label), fold_train_labels_dir)
            else:
                print(f"Uyarı: Etiket dosyası bulunamadı - {label} (train)")
        #kod parçası, belirli bir görüntü dosyasının eğitim dizininde bulunmadığı durumda, doğrulama dizininde 
        # olup olmadığını kontrol eder ve eğer varsa ilgili fold eğitim dizinine kopyalar. Aynı şekilde, görüntü
        # dosyasına karşılık gelen etiket dosyasını da doğrulama dizininden fold eğitim dizinine kopyalar. 
        elif os.path.exists(os.path.join(val_images_dir, image)):
            shutil.copy(os.path.join(val_images_dir, image), fold_train_images_dir)
            if os.path.exists(os.path.join(val_labels_dir, label)):
                shutil.copy(os.path.join(val_labels_dir, label), fold_train_labels_dir)
            else:
                print(f"Uyarı: Etiket dosyası bulunamadı - {label} (val)")
#bir önceki kod parçasının yaptığı işlemleri doğrulama (validation) veri seti için yapar. 
    for idx in val_idx:
        image = images[idx]
        label = os.path.splitext(image)[0] + '.txt'
        if os.path.exists(os.path.join(train_images_dir, image)):
            shutil.copy(os.path.join(train_images_dir, image), fold_val_images_dir)
            if os.path.exists(os.path.join(train_labels_dir, label)):
                shutil.copy(os.path.join(train_labels_dir, label), fold_val_labels_dir)
            else:
                print(f"Uyarı: Etiket dosyası bulunamadı - {label} (train)")
        elif os.path.exists(os.path.join(val_images_dir, image)):
            shutil.copy(os.path.join(val_images_dir, image), fold_val_images_dir)
            if os.path.exists(os.path.join(val_labels_dir, label)):
                shutil.copy(os.path.join(val_labels_dir, label), fold_val_labels_dir)
            else:
                print(f"Uyarı: Etiket dosyası bulunamadı - {label} (val)")

    print(f'Fold {fold} dataset created.')

# Her fold için data.yaml dosyalarını oluştur
for fold in range(k):
    data_yaml = f"data_fold_{fold}.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"train: {os.path.abspath(f'dataset/images/train_fold_{fold}')}\n")
        f.write(f"val: {os.path.abspath(f'dataset/images/val_fold_{fold}')}\n")
        f.write(f"nc: 16\n")
        f.write(f"names:\n")
        for i in range(16):
            f.write(f"  {i}: class_name_{i}\n")
    print(f'Data YAML file for fold {fold} created.')
