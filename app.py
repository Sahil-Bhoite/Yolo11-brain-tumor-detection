import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def load_model(model_path='model.pt'):
    """Load the YOLO model."""
    return YOLO(model_path)

def detect_tumor(model, image):
    """Perform tumor detection on the input image."""
    results = model(image)
    return results

def main():
    # App title and description
    st.title("Brain Tumor Detection")
    st.write("Upload your MRI scan using the sidebar to detect tumors.")
    
    # Sidebar setup
    st.sidebar.header("Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    if uploaded_file is not None:
        try:
            # Read and process the uploaded image
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Display results on the main page
            st.subheader("Uploaded and Detected Images")
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(
                    image, 
                    caption="Uploaded MRI Image", 
                    use_container_width=True  # Updated from use_column_width
                )
            
            if st.sidebar.button("Detect Tumor"):
                with st.spinner("Detecting..."):
                    try:
                        results = detect_tumor(model, image_cv)
                        
                        # Display results in the second column
                        with col2:
                            for result in results:
                                st.image(
                                    result.plot(), 
                                    caption="Detection Results", 
                                    use_container_width=True  # Updated from use_column_width
                                )
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
