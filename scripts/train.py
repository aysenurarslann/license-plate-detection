from ultralytics import YOLO
import os

k = 5

for fold in range(k):
    # Her fold için data.yaml dosyasını yükle
    data_yaml = f"data_fold_{fold}.yaml"

    # YOLO modelini oluştur
    model = YOLO("best.pt")

    # Modeli yeniden eğit
    model.train(data=data_yaml, epochs=50, batch=16, imgsz=640)

    # Modeli kaydet
    model.save(f"runs/detect/train_fold_{fold}/weights/best.pt")

    print(f'Fold {fold} model trained and saved.')
