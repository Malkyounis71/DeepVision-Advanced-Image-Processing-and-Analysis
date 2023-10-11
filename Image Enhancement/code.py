from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import numpy as np
import cv2
import uuid  # Import uuid module
import io
import matplotlib.pyplot as plt

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) settings to allow requests from any origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def apply_brightness_enhancement(image, alpha, beta):
    enhanced_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_img

def apply_contrast_enhancement(image, alpha, beta):
    enhanced_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_img

@app.post("/enhance_brightness/")
async def perform_brightness_enhancement(degree: float = Body(...), image: UploadFile = File(...)):
    try:
        if degree <= 0:
            raise HTTPException(status_code=400, detail="Degree must be greater than 0")
        
        image_data = await image.read()
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        enhanced_image = apply_brightness_enhancement(image_array, alpha=1.0, beta=degree)
        
        # Generate a random name for the output image
        output_filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the enhanced image to the 'output' folder
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path = output_folder / output_filename
        cv2.imwrite(str(output_path), enhanced_image)

        return FileResponse(output_path, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        return {"error": str(e)}

@app.post("/enhance_contrast/")
async def perform_contrast_enhancement(degree: float = Body(...), image: UploadFile = File(...)):
    try:
        if degree <= 0:
            raise HTTPException(status_code=400, detail="Degree must be greater than 0")
        
        image_data = await image.read()
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        enhanced_image = apply_contrast_enhancement(image_array, alpha=degree, beta=0)
        
        # Generate a random name for the output image
        output_filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the enhanced image to the 'output' folder
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path = output_folder / output_filename
        cv2.imwrite(str(output_path), enhanced_image)

        return FileResponse(output_path, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        return {"error": str(e)}
