"""from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
from PIL import Image
import io
import logging
import os
from utils import load_model, preprocess_image, predict, log_prediction

app = FastAPI()

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO)

# Load model
MODEL_PATH = "model/model.pth"

def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Image preprocessing (modify based on your model)
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)

@app.get("/")
def home():
    return {"message": "MRI Tumor Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Logging
    logging.info({
        "prediction": prediction
    })

    return {"prediction": int(prediction)}"""
