import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

app = FastAPI()

def resize_image(image, width, height):
    resized = cv2.resize(image, (width, height))
    return resized

@app.post("/resize-image/")
async def resize_uploaded_image(
    file: UploadFile = File(...),
    new_width: int = 800,
    new_height: int = 600
):
    try:
        # Read the uploaded image
        image = await file.read()
        image_np = np.frombuffer(image, np.uint8)
        img_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Resize the image
        resized_img = resize_image(img_cv2, new_width, new_height)

        # Convert the resized image to bytes
        _, img_encoded = cv2.imencode('.jpg', resized_img)
        resized_bytes = img_encoded.tobytes()

        # Return the resized image as a downloadable file
        return StreamingResponse(io.BytesIO(resized_bytes), media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename=resized_image.jpg"})
    except Exception as e:
        return {"error": str(e)}
