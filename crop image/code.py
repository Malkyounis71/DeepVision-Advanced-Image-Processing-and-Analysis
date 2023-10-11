import numpy as np
import cv2
import os
import uuid
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

app = FastAPI()

# Define the image cropping function
def crop_image(image, x, y, w, h):
    # Crop the image using the given coordinates and size
    output_image = image[y:y+h, x:x+w]
    return output_image

# Define the endpoint for image cropping
@app.post("/crop")
async def crop_image_endpoint(file: UploadFile = File(...), x: int = 0, y: int = 0, w: int = 300, h: int = 300):
    try:
        # Read the image from the uploaded file
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Crop the image using the given parameters
        output_image = crop_image(image, x, y, w, h)

        # Encode the output image as JPEG
        _, output_bytes = cv2.imencode('.jpg', output_image)

        # Generate a random name for the output image
        random_name = str(uuid.uuid4())
        output_path = f'content/{random_name}.jpg'

        # Save the output image with the random name
        with open(output_path, "wb") as output_file:
            output_file.write(output_bytes)

        # Return the cropped image as a streaming response
        return StreamingResponse(open(output_path, "rb"), media_type="image/jpeg")
    except Exception as e:
        print("Error:", e)
        raise e
