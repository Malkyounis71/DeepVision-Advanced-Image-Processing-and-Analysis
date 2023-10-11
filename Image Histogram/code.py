from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import uuid  # Import uuid module
import io
import cv2

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

def calculate_grayscale_histogram(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    return hist

def calculate_color_histogram(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    return hist_b, hist_g, hist_r

@app.post("/calculate_histograms/")
async def perform_histogram_calculation(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        
        # Grayscale Histogram
        hist_gray = calculate_grayscale_histogram(image_data)
        
        # Color Histograms
        hist_b, hist_g, hist_r = calculate_color_histogram(image_data)

        # Generate random names for the output images
        output_filename_gray = str(uuid.uuid4()) + "_gray.jpg"
        output_filename_color = str(uuid.uuid4()) + "_color.jpg"
        
        # Create grayscale histogram plot
        plt.figure(figsize=(8, 6))
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.plot(hist_gray, color='black')
        plt.fill_between(np.arange(256), hist_gray.flatten(), color='blue', alpha=0.5)
        plt.xlim([0, 256])
        plt.grid(True)
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path_gray = output_folder / output_filename_gray
        plt.savefig(output_path_gray)
        plt.close()
        
        # Create color histograms plot
        plt.figure(figsize=(8, 6))
        plt.title('Color Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_r, color='red', label='Red')
        plt.xlim([0, 256])
        plt.legend()
        plt.grid(True)
        output_path_color = output_folder / output_filename_color
        plt.savefig(output_path_color)
        plt.close()

        return {
            "grayscale_histogram": FileResponse(output_path_gray, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename_gray}"}),
            "color_histogram": FileResponse(output_path_color, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename_color}"})
        }
    except Exception as e:
        return {"error": str(e)}
