# =========================================
# STEP 1: INSTALL ULTRALYTICS
# =========================================
!pip install -U ultralytics


# =========================================
# STEP 2: GPU CHECK
# =========================================
import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))


# =========================================
# STEP 3: CREATE DATA YAML
# =========================================
import os

yaml_path = "/kaggle/working/helmet_data.yaml"

yaml_content = """
path: /kaggle/input/helmate-detection

train: train/images
val: valid/images
test: test/images

nc: 2
names:
  - With Helmet
  - Without Helmet
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("YAML file created at:", yaml_path)


# =========================================
# STEP 4: LOAD YOLOv11n MODEL
# =========================================
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # Nano model


# =========================================
# STEP 5: TRAIN MODEL (GPU OPTIMIZED)
# =========================================
model.train(
    data=yaml_path,
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,              # GPU
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    patience=10,
    amp=True,              # Mixed precision (fast on T4)
    project="helmet_training",
    name="yolo11n_helmet",
    exist_ok=True
)


# =========================================
# STEP 6: VALIDATE BEST MODEL
# =========================================
metrics = model.val()
print(metrics)


# =========================================
# STEP 7: CHECK BEST MODEL PATH
# =========================================
best_model_path = "/kaggle/working/helmet_training/yolo11n_helmet/weights/best.pt"

print("Best model exists:", os.path.exists(best_model_path))
print("Best model path:", best_model_path)


# =========================================
# STEP 8: OPTIONAL – TEST INFERENCE
# =========================================
if os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)

    best_model(
        source="/kaggle/input/helmate-detection/test/images",
        conf=0.4,
        save=True
    )


# =========================================
# STEP 9: OPTIONAL – EXPORT MODEL
# =========================================
# best_model.export(format="onnx")

print("✅ Training completed successfully")
