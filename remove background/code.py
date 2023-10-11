from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import os
import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

app = FastAPI()

# Load the DeepLabV3 model
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

# Remove background using the model
def remove_background(model, input_image):
    input_image = Image.open(io.BytesIO(input_image))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    # Create the foreground image with transparency
    b, g, r = cv2.split(np.array(input_image).astype('uint8'))
    a = np.ones(bin_mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    new_mask = np.stack([bin_mask, bin_mask, bin_mask, bin_mask], axis=2)
    foreground = np.where(new_mask, alpha_im, (0, 0, 0, 0)).astype(np.uint8)

    return foreground

# API to remove background and save foreground
@app.post("/remove_background")
async def remove_background_api(image: UploadFile = File(...)):
    try:
        # Load the DeepLabV3 model
        deeplab_model = load_model()

        # Read the uploaded image as bytes
        input_image_data = await image.read()

        # Remove background and save the foreground
        foreground = remove_background(deeplab_model, input_image_data)

        # Create the directory if it doesn't exist
        foreground_dir = "./foreground_images"
        os.makedirs(foreground_dir, exist_ok=True)

        # Save the foreground image to a new file
        foreground_path = f"{foreground_dir}/{os.path.basename(image.filename).replace('.', '_foreground.')}"
        img_fg = Image.fromarray(foreground)

        # Save the image in PNG format to preserve transparency
        img_fg.save(foreground_path, format="PNG")

        # Return the saved foreground image as a response
        return FileResponse(foreground_path)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

