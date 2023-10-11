from fastapi import FastAPI, UploadFile, File
from torchvision import models
from PIL import Image
import torch
import numpy as np
import io
import os
import torchvision.transforms as T

app = FastAPI()


def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(net, img):
    trf = T.Compose([
        T.Resize(640),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inp = trf(img).unsqueeze(0).to(device)
    net = net.to(device)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    return rgb

dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

@app.post("/segment")
async def perform_segmentation(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    segmented_image = segment(dlab, image)

    # Save segmented image
    output_path = os.path.join(output_dir, file.filename)
    Image.fromarray(segmented_image).save(output_path)

    return {"message": "Segmentation completed and saved to 'output' folder.", "output_path": output_path}
