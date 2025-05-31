import streamlit as st

st.set_page_config(page_title="DR Batch Predictor", layout="centered")
st.title("🩺 Diabetic Retinopathy Batch Prediction")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import create_custom_swin_model
from preprocessing import ImagePreprocessor

# Define class names
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = create_custom_swin_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = ImagePreprocessor()
    return model.to(device), device, preprocessor

model, device, preprocessor = load_model_and_preprocessor()

# Streamlit UI
uploaded_files = st.file_uploader("Upload Retinal Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    images = []
    preprocessed_tensors = []

    # Load and preprocess all images
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        images.append((image, image_np))
        tensor = preprocessor.preprocess(image_np).unsqueeze(0)
        preprocessed_tensors.append(tensor)

    # Concatenate into a single batch tensor
    batch_tensor = torch.cat(preprocessed_tensors).to(device)

    # Perform batch prediction
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probs, dim=1)

    # Display predictions
    num_columns = 3
    columns = st.columns(num_columns)

    for idx, ((image, _), pred_class, confidence) in enumerate(zip(images, predicted_classes, confidences)):
        class_name = class_names[pred_class.item()]
        confidence_score = confidence.item() * 100
        col_idx = idx % num_columns
        with columns[col_idx]:
            st.image(image, caption=f"{class_name} ({confidence_score:.2f}%)", width=250)
            st.success(f"Prediction: {class_name} - {confidence_score:.2f}%")

