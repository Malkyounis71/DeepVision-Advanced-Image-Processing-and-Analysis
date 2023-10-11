import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from PIL import Image
import io
import os

app = FastAPI()


def validate_image_format(file_extension: str) -> None:
    supported_formats = ("png", "jpg", "jpeg")
    if file_extension.lower() not in supported_formats:
        raise HTTPException(status_code=400, detail="Unsupported image format. Supported formats are PNG, JPG, and JPEG.")


def validate_direction(direction: str) -> None:
    valid_directions = ("vertical", "horizontal")
    if direction.lower() not in valid_directions:
        raise HTTPException(status_code=400, detail="Invalid direction. Must be either 'vertical' or 'horizontal'.")


def validate_file_size(file_size: int, max_size: int) -> None:
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File size exceeds the maximum allowed size of {max_size} bytes.")


@app.post("/flip")
async def flip_image(file: UploadFile = File(...), direction: str = Body(...)):
    # Validate image format
    validate_image_format(file.filename.split(".")[-1])

    # Validate direction
    validate_direction(direction)

    # Read the uploaded image file
    image_data = await file.read()

    # Validate file size (limit to 5MB, adjust as needed)
    validate_file_size(len(image_data), 5 * 1024 * 1024)

    # Open the image from the binary data
    image = Image.open(io.BytesIO(image_data))

    # Flip the image based on the provided direction
    if direction == "vertical":
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif direction == "horizontal":
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        raise HTTPException(status_code=400, detail="Invalid direction. Must be either 'vertical' or 'horizontal'.")

    # Create the output folder if it doesn't exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Generate a random filename for the flipped image
    random_filename = str(uuid.uuid4())
    output_path = os.path.join(output_folder, f"{random_filename}.png")

    # Save the flipped image to the output folder
    flipped_image.save(output_path, format="PNG")

    return {"message": "Image flipped successfully", "flipped_image_path": output_path}