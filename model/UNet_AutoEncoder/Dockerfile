FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY inferencer.py inferencer.py
COPY UNet_AutoEncoder_UnmaskingModel_train.py UNet_AutoEncoder_UnmaskingModel_train.py
COPY UNet_AutoEncoder_UnmaskingModel.ckpt UNet_AutoEncoder_UnmaskingModel.ckpt

CMD ["uvicorn","inferencer:app","--reload","--host","0.0.0.0","--port","8000"]
