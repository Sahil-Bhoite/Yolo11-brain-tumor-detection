import streamlit as st
from PIL import Image
import numpy as np
import os

def load_model(model_path='model.pt'):
    """Load the YOLO model with error handling."""
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except ImportError:
        st.error("Please install ultralytics: pip install ultralytics")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_tumor(model, image_array):
    """Perform tumor detection with error handling."""
    try:
        results = model(image_array)
        return results
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None

def setup_dependencies():
    """Install required system dependencies."""
    try:
        # Check if running on Linux
        if os.name == 'posix':
            os.system('apt-get update && apt-get install -y libgl1-mesa-glx')
    except Exception as e:
        st.error(f"Error setting up dependencies: {str(e)}")

def main():
    # Setup dependencies
    setup_dependencies()
    
    # App title and description
    st.title("Brain Tumor Detection")
    st.write("Upload your MRI scan using the sidebar to detect tumors.")
    
    # Sidebar setup
    st.sidebar.header("Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Load model with error handling
    model = None
    with st.spinner("Loading model..."):
        try:
            model = load_model()
            if model is None:
                st.warning("Model loading failed. Please check the model file and dependencies.")
                return
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return

    if uploaded_file is not None:
        try:
            # Read and process the uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Display results on the main page
            st.subheader("Uploaded and Detected Images")
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(
                    image, 
                    caption="Uploaded MRI Image", 
                    use_column_width=True
                )
            
            if st.sidebar.button("Detect Tumor"):
                with st.spinner("Detecting..."):
                    try:
                        results = detect_tumor(model, image_array)
                        if results is not None:
                            # Display results in the second column
                            with col2:
                                for result in results:
                                    st.image(
                                        result.plot(), 
                                        caption="Detection Results", 
                                        use_column_width=True
                                    )
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
