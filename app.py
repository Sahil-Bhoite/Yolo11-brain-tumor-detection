import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the trained model
model_path = 'model.pt'   
model = YOLO(model_path)

# Function to perform inference on the uploaded image
def detect_tumor(image):
    results = model(image)
    return results

# Streamlit app layout
st.title("Brain Tumor Detection")
st.write("Upload your MRI scan using the sidebar to detect tumors.")

# Sidebar for uploading the image
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Display results on the main page
    st.subheader("Uploaded and Detected Images")

    # Create columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.sidebar.button("Detect Tumor"):
        with st.spinner("Detecting..."):
            results = detect_tumor(image_cv)

        # Display results in the second column
        with col2:
            for result in results:
                st.image(result.plot(), caption="Detection Results", use_column_width=True)
