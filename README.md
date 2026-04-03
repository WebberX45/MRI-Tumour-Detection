# 🧠 MRI Tumor Detection System (Production-Ready)

Welcome to my **MRI Tumor Detection System** 🚀

This project is a **Deep Learning-based medical imaging system** that detects brain tumors from MRI scans using **Transfer Learning (VGG16)**.

Unlike a basic ML project, this system has been extended into a **production-ready application** with:

- 🌐 Web deployment (Flask)
- 🧠 Deep learning model (TensorFlow/Keras)
- 🔥 Grad-CAM visualization
- 📊 Prediction logging
- 👀 Basic monitoring system
- 🔁 ML lifecycle (retraining-ready pipeline)

---

# 🚀 Features

## 🧠 AI Model
- Built using **VGG16 Transfer Learning**
- Classifies MRI scans into:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor

---

## 🌐 Web Application
- Developed using **Flask**
- User-friendly interface
- Upload MRI images directly
- Displays:
  - Prediction result
  - Uploaded MRI image
  - Grad-CAM heatmap (if tumor detected)

---

## 🔥 Explainability (Grad-CAM)
- Highlights important regions in MRI scans
- Improves trust and interpretability of AI predictions

---

## 📊 Monitoring System
- Logs predictions to:
- - Enables:
- Tracking model predictions
- Detecting data changes
- Supporting future drift detection

---

## 🔁 ML Lifecycle (MLOps Ready)
- Modular pipeline includes:
- `monitor.py` → basic monitoring
- `retrain.py` → retraining pipeline (extendable)
- Designed for automation using cron / workflows


---

# 📊 Dataset

Dataset used:  
👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Contains 4 classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

# 🧠 Model Workflow

1. User uploads MRI image  
2. Image is preprocessed  
3. Model predicts tumor type  
4. Result displayed on UI  
5. Prediction logged for monitoring  
6. (Future) Model retrained with new data  

---

# 📈 Future Improvements

- ✅ Full data drift detection (Evidently)
- ✅ Automated retraining pipeline
- ✅ Model versioning (v1, v2, rollback)
- ✅ FastAPI version for scalability
- ✅ Cloud deployment with GPU
- ✅ Database integration

---

# ⚠️ Limitations

- TensorFlow models are heavy for free hosting
- Monitoring is basic (can be extended)
- Retraining not fully automated yet

---

# 🤝 Contributing

Contributions are welcome!

Steps:
1. Fork the repo  
2. Create a new branch  
3. Submit a pull request  

---

# 💡 Author

Developed by **WebberX45**

---

# ⭐ Final Note

This project demonstrates the evolution from:

👉 **Deep Learning Model → Production ML System (MLOps-ready)**

Combining:
- AI
- Computer Vision
- Web Development
- ML Monitoring

into a real-world deployable application 🚀
