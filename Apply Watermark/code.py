from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import random
import string

app = FastAPI()

# Allow CORS to prevent cross-origin issues
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

def generate_random_filename():
    random_string = ''.join(random.choices(string.ascii_letters, k=10))
    return f"output_{random_string}.png"

def add_watermark(source_image_path, watermark_image_path, x_position, y_position):
    imgS = Image.open(source_image_path).convert("RGBA")
    imgW = Image.open(watermark_image_path)
    imgW = imgW.resize((100, 100))

    imgS.paste(imgW, (x_position, y_position), imgW.convert("RGBA"))

    output_filename = generate_random_filename()
    output_path = os.path.join("output", output_filename)  # Save in the "output" folder
    imgS.save(output_path, "PNG", quality=90)
    return output_path

@app.post("/add_watermark/")
async def add_watermark_to_image(x_position: int = Form(...), y_position: int = Form(...), source_image: UploadFile = File(...), watermark_image: UploadFile = File(...)):
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    source_image_path = os.path.join(temp_dir, source_image.filename)
    watermark_image_path = os.path.join(temp_dir, watermark_image.filename)

    with open(source_image_path, "wb") as f:
        f.write(source_image.file.read())

    with open(watermark_image_path, "wb") as f:
        f.write(watermark_image.file.read())

    output_image_path = add_watermark(source_image_path, watermark_image_path, x_position, y_position)

    os.remove(source_image_path)
    os.remove(watermark_image_path)

    return FileResponse(output_image_path, media_type="image/png")
