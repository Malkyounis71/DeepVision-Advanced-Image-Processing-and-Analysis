from fastapi import FastAPI, UploadFile, File
import os
import io
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

app = FastAPI()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Endpoint to accept image and detect objects
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded image
    image = Image.open(io.BytesIO(await file.read()))

    # Perform inference
    results = model([image])

    # Get the first image's predictions
    predictions = results.xyxy[0]

    # Draw bounding boxes on the image
    annotated_image = results.render()[0]

    # Convert the NumPy array to a PIL Image
    pil_annotated_image = Image.fromarray(annotated_image)

    # Save the annotated image
    output_path = os.path.join("content", file.filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_annotated_image.save(output_path)

    return {"file_saved": output_path, "predictions": predictions.tolist()}
