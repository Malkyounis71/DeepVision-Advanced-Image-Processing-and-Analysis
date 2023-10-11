from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

app = FastAPI()

# Constants for image validation
MAX_IMAGE_DIMENSION = 1024  # Maximum allowed image dimension (both width and height)
MAX_FILE_SIZE = 2 * 1024 * 1024  # Maximum allowed file size (2 MB)

# Load the DeepLabV3 model
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

# Remove background using the model
def remove_background(model, input_image):
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

# Define the class labels
class_labels = ["background", "foreground"]

# Load the pre-trained ResNet-18 model
def load_classification_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_labels))
    model.eval()
    return model

@app.post("/classify/")
async def classify_image(file: UploadFile):
    if not file.content_type.startswith('image'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are accepted.")

    # Read and preprocess the uploaded image
    image_data = await file.read()
    input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Validate image dimensions
    if max(input_image.size) > MAX_IMAGE_DIMENSION:
        raise HTTPException(status_code=400, detail=f"Image dimensions exceed the maximum allowed ({MAX_IMAGE_DIMENSION}px).")

    # Validate file size
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds the maximum allowed ({MAX_FILE_SIZE} bytes).")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Load the models
    model = load_model()
    classification_model = load_classification_model()

    # Remove background
    foreground = remove_background(model, input_image)

    # Perform classification
    with torch.no_grad():
        output = classification_model(input_batch)

    predicted_class_idx = torch.argmax(output).item()
    predicted_class_label = class_labels[predicted_class_idx]

    return JSONResponse(content={"predicted_class": predicted_class_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

