from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
from pathlib import Path
import numpy as np
import uuid  # For generating random names
from PIL import Image
import io

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

def convert_color_space(image_data, color_space):
    original_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    if color_space == 'RGB':
        converted_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        converted_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        converted_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError("Invalid color space selection. Available options are: RGB, HSV, LAB")
    
    return converted_image

@app.post("/convert_color_space/")
async def perform_color_space_conversion(image: UploadFile = File(...), color_space: str = Form(...)):
    try:
        image_data = await image.read()
        converted_image = convert_color_space(image_data, color_space)

        # Generate a random name for the output image
        output_filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the converted image to the 'output' folder
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path = output_folder / output_filename
        cv2.imwrite(str(output_path), converted_image)

        return FileResponse(output_path, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        return {"error": str(e)}
