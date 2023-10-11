from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib.pyplot as plt
import random
import string
import os
from pathlib import Path
import io

app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing) to allow requests from any domain
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Function to generate a random string for image filenames
def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Function to binarize the image
def binarize_image(input_image: Image.Image, threshold: int) -> Image.Image:
    gray_image = input_image.convert('L')
    binary_image = gray_image.point(lambda p: 255 if p > threshold else 0)
    return binary_image

# Route to binarize image
@app.post("/binarize_image/")
async def binarize_image_endpoint(threshold: int = Form(...), image: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))

        # Binarize the image
        result_image = binarize_image(image_pil, threshold)

        # Generate a random filename and save the result image
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)
        output_filename = f"{random_string()}.png"
        output_path = output_folder / output_filename
        result_image.save(output_path)

        return {"message": "Image binarization successful", "output_filename": output_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

