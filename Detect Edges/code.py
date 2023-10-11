from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import numpy as np
import cv2
import uuid  # Import uuid module
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

def perform_edge_detection(image_data, edge_detection_type):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
    
    if edge_detection_type.lower() == 'x':
        edge_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    elif edge_detection_type.lower() == 'y':
        edge_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    elif edge_detection_type.lower() == 'xy':
        edge_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    else:
        raise HTTPException(status_code=400, detail="Invalid edge detection type selection.")
    
    return edge_img

@app.post("/perform_edge_detection/")
async def edge_detection(image: UploadFile = File(...), edge_detection_type: str = Body(...)):
    try:
        image_data = await image.read()
        edge_img = perform_edge_detection(image_data, edge_detection_type)

        # Generate a random name for the output image
        output_filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the edge detection result to the 'output' folder
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_path = output_folder / output_filename
        cv2.imwrite(str(output_path), edge_img)

        return FileResponse(output_path, media_type='image/jpeg', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    except Exception as e:
        return {"error": str(e)}
