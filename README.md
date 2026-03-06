# Brain Tumor Detection System (MRI Analysis)

Welcome to my **Brain Tumor Detection System** project. In this project, I developed a **Deep Learning-based medical imaging system** that detects the presence of brain tumors from MRI scans.

The system uses **Transfer Learning with the VGG16 Convolutional Neural Network** to classify MRI images into different tumor categories. The trained model is then deployed through a **Flask web application** that allows users to upload MRI images and receive instant predictions.

Additionally, the system generates a **Grad-CAM heatmap** that highlights the region of the image responsible for the prediction, improving model interpretability.

---

## Dataset

The dataset used in this project is the **Brain Tumor MRI Dataset**, available on Kaggle:

<a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset">Brain Tumor MRI Dataset</a>

This dataset contains MRI scan images categorized into four classes:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The dataset is organized into two main directories:

- **Training Dataset**
- **Testing Dataset**

Each folder contains labeled MRI scan images that are used to train and evaluate the deep learning model.

The dataset enables the development of **AI-based medical imaging systems for brain tumor detection and classification**.

---

## Tools & Technologies

### Programming
- Python

### Deep Learning Libraries
- TensorFlow
- Keras
- NumPy
- OpenCV

### Data Visualization
- Matplotlib
- Seaborn

### Web Development
- Flask
- HTML
- CSS
- Bootstrap

---

## Project Workflow

### 1. Data Preparation
- Extracted MRI dataset
- Organized images into training and testing folders
- Loaded images using TensorFlow image datasets
- Performed label encoding and dataset shuffling

### 2. Data Visualization
- Displayed sample MRI images
- Analyzed class distribution
- Created plots showing tumor type distribution

### 3. Data Augmentation
Applied preprocessing techniques to improve model generalization:

- Brightness adjustment
- Contrast enhancement
- Image normalization

---

### 4. Model Architecture

The model uses **Transfer Learning with the VGG16 architecture**.

Key components:

- Pretrained VGG16 convolutional backbone
- Flatten layer
- Dense layers
- Dropout layers
- Softmax output layer

This architecture allows the model to extract complex image features from MRI scans.

---

### 5. Model Training

The model was trained using:

- **Adam optimizer**
- **Sparse categorical cross-entropy loss**
- **Batch training with image generators**

Performance was evaluated using:

- Accuracy
- Loss curves
- Confusion Matrix
- ROC Curve

---

### 6. Model Deployment

The trained model was saved as:

```
model.h5
```

A **Flask web application** was developed to allow users to upload MRI scans and receive predictions.

Backend implementation example: :contentReference[oaicite:1]{index=1}

---

### 7. Grad-CAM Visualization

To improve model explainability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is implemented.

Grad-CAM highlights the **important regions of the MRI scan that influenced the model's prediction**, helping doctors visually interpret results.

---

## Web Application Features

The web interface allows users to:

- Upload MRI images
- Detect whether a tumor is present
- Identify tumor type
- View Grad-CAM heatmap visualization

Frontend interface example: :contentReference[oaicite:2]{index=2}  
Styling implementation: :contentReference[oaicite:3]{index=3}

---

## Key Insights

- Deep learning models can effectively detect patterns in medical imaging.
- Transfer learning significantly improves performance when training data is limited.
- Grad-CAM visualization helps make deep learning models **more interpretable for medical applications**.
- CNN architectures are highly effective for **medical image classification tasks**.

---

## Real-World Applications

This system demonstrates practical applications of AI in healthcare:

### Medical Diagnosis Support
Assist doctors in detecting brain tumors from MRI scans.

### Radiology Automation
Help radiologists analyze large numbers of scans efficiently.

### Early Disease Detection
Identify tumors at early stages for faster treatment.

### AI-powered Healthcare Systems
Integrate into hospital diagnostic systems.

### Medical Research
Assist researchers studying tumor classification patterns.

---

## Conclusion

This project demonstrates a **complete deep learning pipeline for medical image analysis**, including:

- Data preprocessing
- Transfer learning with CNNs
- Model evaluation
- Explainable AI (Grad-CAM)
- Web deployment using Flask

By combining **AI, computer vision, and web development**, the system provides a powerful tool for assisting brain tumor detection from MRI images.

---

## Contact

For questions, feedback, or collaboration:

LinkedIn:  
https://www.linkedin.com/in/pranav-kumar-553583394  

Email:  
d.sci.pranav@gmail.com
