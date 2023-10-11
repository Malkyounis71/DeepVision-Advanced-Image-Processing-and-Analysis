import os
import uuid
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI()

@app.post("/rotate_image/")
async def rotate_image(image: UploadFile = File(...), degree: int = Form(...)):
    # Validate degree value
    if degree not in [0, 90, 180, 270]:
        return {"error": "Invalid degree value. Please choose from 0, 90, 180, or 270."}
    
    # Create the output folder if it doesn't exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Generate a unique filename for the rotated image
        filename = str(uuid.uuid4()) + ".jpg"
        
        # Save the uploaded image to a temporary file
        temp_file_path = os.path.join(output_folder, "temp.jpg")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await image.read())
        
        # Open the temporary file using PIL
        with Image.open(temp_file_path) as img:
            # Rotate the image based on the degree value
            rotated_img = img.rotate(degree, expand=True)
            
            # Save the rotated image to the output folder with the generated filename
            output_path = os.path.join(output_folder, filename)
            rotated_img.save(output_path)
        
        # Delete the temporary file
        os.remove(temp_file_path)
        
        return {"message": "Image rotated successfully.", "output_path": output_path}
    
    except Exception as e:
        return {"error": str(e)}