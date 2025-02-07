import cv2
import os
from ultralytics import YOLO

k = 5
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for fold in range(k):
    # Eğitilmiş modeli yükle
    model = YOLO(f"runs/detect/train_fold_{fold}/weights/best.pt")

    # Resimleri yükle
    input_images_dir = f'dataset/images/val_fold_{fold}'
    output_dir = f'output_fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_images_dir):
        if image_name.lower().endswith(valid_extensions):
            image_path = os.path.join(input_images_dir, image_name)
            image = cv2.imread(image_path)
            
            # Plaka tespiti yap
            results = model(image)
            
            # Sonuçları kaydet
            output_image_path = os.path.join(output_dir, image_name)
            for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cv2.imwrite(output_image_path, image)

    print(f'Fold {fold} detection completed and results saved.')
