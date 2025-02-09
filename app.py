import streamlit as st
import sys
import traceback
try:
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"Python version: {sys.version}")
    st.error(traceback.format_exc())
    st.stop()

# Configure page
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":medical_symbol:",
    layout="wide"
)

# Load the trained model
try:
    model_path = 'model.pt'
    model = YOLO(model_path)
except Exception as model_load_error:
    st.error(f"Error loading model: {model_load_error}")
    st.stop()

def detect_tumor(image):
    try:
        results = model(image)
        return results
    except Exception as inference_error:
        st.error(f"Error during tumor detection: {inference_error}")
        return None

def main():
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("### AI-Powered MRI Scan Analysis")
    
    # Sidebar for uploading the image
    st.sidebar.header("🖼️ Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan for tumor detection"
    )
    
    # Create columns for side-by-side display
    if uploaded_file is not None:
        try:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create two columns for side-by-side image display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Uploaded MRI Scan")
                st.image(image, caption="Original MRI Image", use_column_width=True)
            
            # Tumor Detection Button in sidebar
            if st.sidebar.button("🔍 Detect Tumor", key="detect_button"):
                with st.spinner("Analyzing image..."):
                    results = detect_tumor(image_cv)
                    
                    if results is not None:
                        with col2:
                            st.subheader("🩺 Detection Results")
                            for result in results:
                                # Plot detected regions
                                annotated_image = result.plot()
                                st.image(
                                    annotated_image, 
                                    caption="Tumor Detection Visualization", 
                                    use_container_width=True
                                )
                                
                                # Display detection statistics with improved formatting
                                st.write("### Detection Summary")
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    st.write(f"- Detected: Brain Tumor")
                                    st.write(f"- Detection Confidence: {conf:.2%}")
                    else:
                        st.error("No results could be generated. Please check the input image.")
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.error(traceback.format_exc())
    
    # Information section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About this App**\n"
        "- Uses AI to detect brain tumors from MRI scans\n"
        "- Powered by Ultralytics YOLO\n"
        "- Provides detection confidence and visualization"
    )

if __name__ == "__main__":
    main()
