import os
import io
import time
import random
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = FastAPI()

# Load the ESRGAN model
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)

def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.BICUBIC)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

def enhance_resolution(lr_image):
    fake_image = model(lr_image)
    fake_image = tf.squeeze(fake_image)
    return fake_image

def generate_random_filename():
    random_name = f"output_{int(time.time())}_{random.randint(0, 1000)}.jpg"
    return random_name

def save_image_to_file(image, filename):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    output_path = os.path.join("output", filename)
    image.save(output_path)
    return output_path

@app.post("/enhance_resolution/")
async def enhance_resolution_api(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

        lr_image = preprocess_image(image)
        fake_image = enhance_resolution(lr_image)

        output_filename = generate_random_filename()
        output_path = save_image_to_file(fake_image, output_filename)

        return FileResponse(output_path, media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

if not os.path.exists("output"):
    os.mkdir("output")
