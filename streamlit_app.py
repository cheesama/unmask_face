from PIL import Image

import torchvision.transforms as transforms
import streamlit as st
import numpy as np
import onnx
import onnxruntime
import os, sys

# Set page title
st.title("Unmask Face Inferencer")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def valid_transform(image, img_size=256):
    img_tensor = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )(image).unsqueeze(0)

    return img_tensor

def denormalize(img_tensor):
    return (img_tensor * 0.5) + 0.5

# Load pix2pix model
@st.cache(allow_output_mutation=True)
def load_pix2pix_unmasking_model():
    with st.spinner("Loading unmasking model..."):
        model = onnxruntime.InferenceSession('pix2pix_generator.onnx')

        return model
    
if __name__ == "__main__":
    ## optical character recognition
    st.subheader("Unmasking Face Image Generation")
    model = load_pix2pix_unmasking_model()
    uploaded_file = st.file_uploader("Upload Image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        ort_inputs = {model.get_inputs()[0].name: to_numpy(valid_transform(image))}
        output = model.run(None, ort_inputs)
        output = output[0]
        #output = denormalize(output)

        st.image(output.squeeze(0).transpose(1,2,0))


        

