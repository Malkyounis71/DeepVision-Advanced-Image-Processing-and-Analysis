import cv2
import numpy as np
import os
import random
import string
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Tuple
from pathlib import Path

app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing) to allow requests from any domain
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Function to generate a random string for image filenames
def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Function to perform image alignment
def align_images(base: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, str]:
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create()
    base_keys, base_desc = detector.detectAndCompute(base_gray, None)
    curr_keys, curr_desc = detector.detectAndCompute(curr_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(base_desc, curr_desc)

    min_dist = min(match.distance for match in matches)
    good_matches = [match for match in matches if match.distance <= 3 * min_dist]

    base_matches = np.array([base_keys[match.queryIdx].pt for match in good_matches])
    curr_matches = np.array([curr_keys[match.trainIdx].pt for match in good_matches])

    transformation, hom_status = cv2.findHomography(curr_matches, base_matches, cv2.RANSAC, 1)

    height, width = curr.shape[:2]
    mod_photo = cv2.warpPerspective(curr, transformation, (width, height))

    mod_photo_resized = cv2.resize(mod_photo, (base.shape[1], base.shape[0]))

    new_image = cv2.addWeighted(mod_photo_resized, 0.5, base, 0.5, 1)

    # Generate a random filename and save the result image
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)
    output_filename = f"{random_string()}.jpg"
    output_path = output_folder / output_filename
    cv2.imwrite(str(output_path), new_image)

    return new_image, output_filename

# Route to perform image alignment and get the result
@app.post("/align_and_get_result/")
async def align_and_get_result(base_image: UploadFile = File(...), curr_image: UploadFile = File(...)):
    try:
        # Read uploaded images
        base_data = await base_image.read()
        curr_data = await curr_image.read()
        base_np = np.frombuffer(base_data, np.uint8)
        curr_np = np.frombuffer(curr_data, np.uint8)
        base_img = cv2.imdecode(base_np, cv2.IMREAD_COLOR)
        curr_img = cv2.imdecode(curr_np, cv2.IMREAD_COLOR)

        # Perform image alignment and get the aligned image
        result_image, output_filename = align_images(base_img, curr_img)

        return {"message": "Image alignment successful", "output_filename": output_filename, "result_image": result_image.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

