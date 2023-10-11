from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import skimage.filters as filters
import scipy.ndimage
import io

app = FastAPI()

def apply_filter(img, filter_type):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if filter_type == 'gaussian':
        smoothed = cv2.GaussianBlur(gray, (95, 95), 0)
        result = cv2.addWeighted(gray, 1.5, smoothed, -0.5, 0)
    elif filter_type == 'ndimage':
        result = scipy.ndimage.gaussian_filter(gray, sigma=3)
    elif filter_type == 'median':
        result = scipy.ndimage.median_filter(gray, size=3)
    elif filter_type == 'sharpen':
        smooth = cv2.GaussianBlur(gray, (95, 95), 0)
        division = cv2.divide(gray, smooth, scale=40)
        result = filters.unsharp_mask(division, radius=2.5, amount=1)
        result = (220 * result).clip(0, 220).astype(np.uint8)
    elif filter_type == 'edges':
        result = cv2.Canny(gray, 100, 200)
    else:
        raise ValueError("Invalid filter type")
    
    return result

@app.post("/apply_filter/")
async def apply_filter_route(filter_type: str, image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Apply the selected filter
        filtered_img = apply_filter(img, filter_type)
        
        # Convert the filtered image to bytes
        _, img_encoded = cv2.imencode('.jpg', filtered_img)
        img_bytes = img_encoded.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}
