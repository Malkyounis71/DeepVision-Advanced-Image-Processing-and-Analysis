from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from urllib.request import urlretrieve
import tempfile
import shutil
import cv2

app = FastAPI()

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    # Split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # Add an alpha channel and fill with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # Merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # Create a transparent background
    bg = np.zeros(alpha_im.shape)
    # Setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # Copy only the foreground color pixels from the original image where the mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground

def remove_background(model, input_file):
    input_image = Image.open(input_file)
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

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask

def custom_background(background_file, foreground):
    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)
    x = (background.size[0] - final_foreground.size[0]) / 2 + 0.5
    y = (background.size[1] - final_foreground.size[1]) / 2 + 0.5
    box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
    crop = background.crop(box)
    final_image = crop.copy()
    # Put the foreground in the center of the background
    paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
    final_image.paste(final_foreground, paste_box, mask=final_foreground)
    return final_image

@app.post("/replace_background/")
async def replace_background(file: UploadFile = UploadFile(...), background: UploadFile = UploadFile(...)):
    try:
        # Save the uploaded images to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input_image:
            shutil.copyfileobj(file.file, temp_input_image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_background_image:
            shutil.copyfileobj(background.file, temp_background_image)

        # Load the DeepLabV3 model
        deeplab_model = load_model()
        if deeplab_model is None:
            raise HTTPException(status_code=500, detail="Failed to load the model")

        # Perform background replacement
        foreground, bin_mask = remove_background(deeplab_model, temp_input_image.name)
        final_image = custom_background(temp_background_image.name, foreground)

        # Specify the folder where the image will be saved
        server_images_folder = './server_images/'

        # Save the result to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir=server_images_folder, suffix=".jpg") as temp_output:
            final_image.save(temp_output.name)

        # Return the resulting image
        return FileResponse(temp_output.name, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=final.jpg"})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
