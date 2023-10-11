from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import numpy as np
import cv2
import uuid  # Import uuid module

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

def denoise_image(image_data):
    noisy_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    
    # Apply non-local means denoising
    denoised_image_rgb = cv2.fastNlMeansDenoisingColored(noisy_image_rgb, None, 10, 10, 7, 21)
    
    return denoised_image_rgb

@app.post("/denoise_image/")
async def perform_image_denoising(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        denoised_image = denoise_image(image_data)

        # Generate a random name for the output image
        output_filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the denoised image to the 'output' folder
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path = output_folder / output_filename
        cv2.imwrite(str(output_path), cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))

        return FileResponse(output_path, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        return {"error": str(e)}
