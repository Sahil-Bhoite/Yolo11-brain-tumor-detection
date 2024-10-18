# YOLOv11 Brain Tumor Detection

## Overview
This project implements an advanced brain tumor detection system using the latest YOLOv11 model. Trained on a carefully curated dataset, the model achieves an impressive accuracy of over 95% in detecting brain tumors from MRI scans, demonstrating the potential of deep learning in medical imaging diagnostics.

## Features
- Utilizes the state-of-the-art YOLOv11 object detection model
- High accuracy (>95%) in tumor detection
- User-friendly interface built with Streamlit
- Real-time detection on uploaded MRI scans

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/Sahil-Bhoite/Yolo11-brain-tumor-detection.git
   cd Yolo11-brain-tumor-detection
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

3. Use the sidebar to upload an MRI scan image.

4. Click the "Detect Tumor" button to process the image and view the results.

## Model Details
- Architecture: YOLOv11
- Training Dataset: Custom dataset of brain MRI scans
- Accuracy: >95%
- File: `model.pt` (not included in the repository due to size constraints)

## Project Structure
- `app.py`: Main Streamlit application
- `model.pt`: Trained YOLOv11 model (not included in repo)
- `data.yaml`: Dataset configuration file
- `Training.ipynb`: Jupyter notebook used for model training
- `test/`, `train/`, `valid/`: Directories containing dataset images
- `Deep Learning Project Report.pdf`: Detailed project report

## Contributing
Contributions to improve the model accuracy, expand the dataset, or enhance the user interface are welcome. Please feel free to fork the repository and submit pull requests.

## License
[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [Streamlit](https://streamlit.io/) for the web application framework

## Contact
For any queries or assistance regarding this project, please feel free to reach out:

Sahil Bhoite - [LinkedIn](https://www.linkedin.com/in/sahil-bhoite/)

Project Link: [https://github.com/Sahil-Bhoite/Yolo11-brain-tumor-detection](https://github.com/Sahil-Bhoite/Yolo11-brain-tumor-detection)

## Future Improvements
- Expand the training dataset to improve model robustness
- Implement multi-class detection for different types of brain tumors
- Optimize the model for deployment on edge devices for real-time detection in clinical settings
