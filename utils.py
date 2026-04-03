import torch
import numpy as np
from PIL import Image
import io
import csv
import os

# Load model
def load_model(model_path="model/model.pth"):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)

# Predict
def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# Evaluate model
def evaluate(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# Log predictions
def log_prediction(prediction, file_path="data/production_data.csv"):
    os.makedirs("data", exist_ok=True)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([prediction])

# Label mapping
CLASS_NAMES = ["No Tumor", "Tumor"]

def get_label(prediction):
    return CLASS_NAMES[prediction]