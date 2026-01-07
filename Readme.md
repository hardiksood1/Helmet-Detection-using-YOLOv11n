
# 🪖 Helmet Detection using YOLOv11n

## 📌 Overview
This repository demonstrates an **end-to-end Helmet Detection system** built using **YOLOv11n**.  
The project covers **training logic, validation flow, and real-time video inference** to identify whether a rider is **wearing a helmet or not**.

⚠️ **Important:**  
This project is created **for demo and educational purposes only**.

---

## 📂 Training Files Notice
🚫 **Training data and raw training files are NOT uploaded to this repository.**

Reason:
- Dataset size is large
- Dataset may contain licensed or sensitive content
- Repository is intended to showcase **pipeline, code structure, and results**, not dataset distribution

You can train the model by plugging in **your own dataset** following the YOLO format described below.

---

## 🚧 Real-World Deployment Disclaimer
For **real-time, production-grade deployment**, significantly more data is required.

📊 **Recommended Dataset Size**
- **At least 5,000 images per class**
- Multiple angles (front, side, back, tilted)
- Different lighting conditions (day, night, rain)
- Diverse camera resolutions and distances

> ⚠️ *Zero hallucination or 100% accuracy cannot be guaranteed in computer vision systems.  
Large-scale, diverse data is essential to approach production reliability.*

---

## 🎯 Key Features
- ✅ YOLOv11n (Nano) model for fast inference
- ✅ Two-class detection: Helmet / No Helmet
- ✅ GPU-optimized training pipeline (code provided)
- ✅ Real-time video detection with visual alerts
- ✅ Clean, modular demo code

---

## 🗂️ Repository Structure
```
├── helmet_data.yaml        # Dataset configuration template
├── helmet.py              # Training script (dataset not included)
├── test.py                # Video inference script
├── models/
│   └── best.pt            # Demo trained weights
├── outputs/
│   └── output2.mp4        # Inference result
└── README.md
```

---

## 🧠 Dataset Format (Required)
```
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

---

## ⚙️ Training Configuration (Demo)
- Model: `yolo11n.pt`
- Epochs: `40`
- Image Size: `640`
- Batch Size: `16`
- Optimizer: `AdamW`
- Mixed Precision: Enabled (AMP)
- Hardware: NVIDIA T4 GPU

---

## ▶️ Inference Logic
- 🟢 Green box → Helmet
- 🔴 Red box → No Helmet
- 🚨 Alert banner for safety violation

---

## 🚀 How to Run

### Run Inference Only
```bash
python test.py
```

### Train with Your Own Dataset
```bash
python helmet.py
```

---

## 📦 Dependencies
```bash
pip install ultralytics opencv-python torch
```

---

## 🛡️ Intended Use
- Proof-of-concept demonstrations
- Academic learning
- Computer vision pipeline reference
- Not production-ready without retraining

---

## 👨‍💻 Author
**Hardik Sood**  
MSc Data Science | Computer Vision & AI Systems  

---

## ⭐ Final Note
This repository focuses on **code quality, pipeline clarity, and deployment logic**.

For real-world use:
- Upload your own dataset
- Retrain the model
- Perform field-specific validation
