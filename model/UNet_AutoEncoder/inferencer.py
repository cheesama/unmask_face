from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from UNet_AutoEncoder_UnmaskingModel_train import UnmaskingModel

import io
import pytorch_lightning as pl

# load checkpoint
model = None

app = FastAPI()

model = UnmaskingModel.load_from_checkpoint('UNet_AutoEncoder_UnmaskingModel.ckpt')
#img_size = model.img_size
img_size=128
prediction_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
    
@app.post("/unmasking", response_class=FileResponse)
async def create_file(file: bytes = File(...)):
    img = Image.open(io.BytesIO(file)).convert('RGB')
    img = prediction_transform(img)
    if len(img.size()) < 4:
        img = img.unsqueeze(0)

    prediction = to_pil_image(model.predict(img).squeeze(0))
    prediction.save('unmasked.png')

    return FileResponse('unmasked.png')