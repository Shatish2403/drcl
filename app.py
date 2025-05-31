import streamlit as st

st.set_page_config(page_title="DR Grad-CAM Viewer", layout="centered")
st.title("🩺 Diabetic Retinopathy Grad-CAM Viewer")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import create_custom_swin_model
from preprocessing import ImagePreprocessor

# Define class names
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Define reshape_transform for GradCAM
def reshape_transform(tensor, height=8, width=8):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = create_custom_swin_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = ImagePreprocessor()
    return model, device, preprocessor

model, device, preprocessor = load_model_and_preprocessor()

# Streamlit UI
uploaded_files = st.file_uploader("Upload Retinal Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    num_columns = 3  # Adjust this for the number of columns you want in your grid
    columns = st.columns(num_columns)

    # Loop through each uploaded image and process them
    for idx, uploaded_file in enumerate(uploaded_files):
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Select the column in the grid
        col_idx = idx % num_columns  # Ensure images fill up columns sequentially
        with columns[col_idx]:
            st.image(image, caption=f"Uploaded Image {idx+1}", width=250)

            # Process each image when the respective button is clicked
            if st.button(f"🔍 Predict & Show Grad-CAM {idx+1}", key=f"gradcam_button_{idx}"):
                with st.spinner("Processing..."):
                    # Preprocess image
                    preprocessed_tensor = preprocessor.preprocess(image_np).unsqueeze(0).to(device)

                    # Predict
                    with torch.no_grad():
                        output = model(preprocessed_tensor)
                        probs = F.softmax(output, dim=1)
                        confidence, predicted_class = torch.max(probs, dim=1)
                        class_name = class_names[predicted_class.item()]
                        confidence_score = confidence.item() * 100

                    # Grad-CAM
                    target_layers = [model.layers[-1].blocks[-1].norm2]
                    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                    grayscale_cam = cam(input_tensor=preprocessed_tensor, targets=None)[0, :]
                    grayscale_cam_resized = cv2.resize(grayscale_cam, (256, 256))
                    rgb_img = np.float32(image_np) / 255.0
                    rgb_img_resized = cv2.resize(rgb_img, (256, 256))
                    cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam_resized, use_rgb=True)

                # Display Grad-CAM image
                st.image(cam_image, caption=f"Grad-CAM: {class_name} ({confidence_score:.2f}%)", width=200)
                st.success(f"Prediction: {class_name} with confidence {confidence_score:.2f}%")

