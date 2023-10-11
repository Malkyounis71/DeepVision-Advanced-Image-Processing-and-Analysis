from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from typing import List
import io
import os
import uuid

app = FastAPI()

class ImageStitcher:
    def stitch(self, images):
        imgs = []
        for img_bytes in images:
            img = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs.append(img)
        
        stitcher = cv2.Stitcher_create()
        status, result = stitcher.stitch(imgs)
        
        if status == cv2.Stitcher_OK:
            return result
        else:
            return None

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

@app.post("/stitch")
async def stitch_images(files: List[UploadFile] = File(...)):
    image_bytes_list = [await file.read() for file in files]
    
    stitcher = ImageStitcher()
    stitched_image = stitcher.stitch(image_bytes_list)
    
    if stitched_image is not None:
        output_name = str(uuid.uuid4()) + ".jpg"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, stitched_image)

        return StreamingResponse(io.BytesIO(stitched_image), media_type="image/jpeg")
    else:
        return {"message": "Stitching was not successful."}
