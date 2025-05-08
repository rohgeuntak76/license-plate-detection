# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

import requests
import json

class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit and Ultralytics YOLO models.

    Attributes:
        st (module): Streamlit module for UI creation.
        temp_dict (dict): Temporary dictionary to store the model path and other configuration.
        model_path (str): Path to the loaded model.
        model (YOLO): The YOLO model instance.
        source (str): Selected video source (webcam or video file).
        enable_trk (str): Enable tracking option ("Yes" or "No").
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for non-maximum suppression.
        org_frame (Any): Container for the original frame to be displayed.
        ann_frame (Any): Container for the annotated frame to be displayed.
        vid_file_name (str | int): Name of the uploaded video file or webcam index.
        selected_ind (List[int]): List of selected class indices for detection.

    Methods:
        web_ui: Sets up the Streamlit web interface with custom HTML elements.
        sidebar: Configures the Streamlit sidebar for model and inference settings.
        source_upload: Handles video file uploads through the Streamlit interface.
        configure: Configures the model and loads selected classes for inference.
        inference: Performs real-time object detection inference.

    Examples:
        >>> inf = Inference(model="path/to/model.pt")  # Model is an optional argument
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        import streamlit as st

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file_name = None  # Video file name or webcam index
        self.vid_file = None
        self.file_bytes = None
        self.selected_ind = []  # List of selected class indices for detection
        self.selected_classes = []
        self.model = None  # YOLO model instance

        self.temp_dict = {"model": None, **kwargs}
        self.model_dir = None
        self.model_path = None  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="YOLO Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Data Type",
            ("Image", "Video"),
        )  # Add source selection dropdown
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        # self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        self.vid_file_name = ""
        if self.source == "Video":
            self.vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if self.vid_file is not None:
                self.org_frame.video(self.vid_file,muted=True)    
        elif self.source == "Image":
            self.vid_file = self.st.sidebar.file_uploader("Upload Image File", type=["jpeg", "png"])
            if self.vid_file is not None:
                self.org_frame.image(self.vid_file)
            
        # if self.vid_file is not None:
        #     self.file_bytes = io.BytesIO(self.vid_file.read())  # BytesIO Object

    def configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection
        available_models = ["Yolo11s","Yolo11s_20epochs_best"]
        # if self.model_path:  # If user provided the custom model, insert model without suffix as *.pt is added later
        #     available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model_dir = "../inference/models"
            self.model = YOLO(f"{self.model_dir}/{selected_model.lower()}.pt")  # Load the YOLO model
            class_names = list(self.model.names.values())  # Convert dictionary to list of class names
            self.model_load = self.st.success("Model loaded successfully!")
        
        # Multiselect box with class names and get indices of selected classes
        if len(class_names) > 1:
            self.selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=[class_names[i] for i in [2,3,5,7]])   
        else:
            self.selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in self.selected_classes]

        if not isinstance(self.selected_ind, list):  # Ensure selected_options is a list
            self.selected_ind = list(self.selected_ind)

    def display_frame(self,frame_bytes):
        # Convert bytes to image
        # image = Image.open(io.BytesIO(frame_bytes))
        image = io.BytesIO(frame_bytes)
        # Update the placeholder
        self.ann_frame.image(image)
        # self.org_frame.image(image)

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app
        if self.st.sidebar.button("Start"):
            self.model_load.empty()   
            stop_button = self.st.button("Stop")  # Button to stop the inference
            if self.source == "Image":
                print(self.selected_classes)
                if self.selected_classes[0] == 'License_Plate':
                    url = "http://localhost:8000/api/image/plates/detect/annotated"
                    files = {
                        'image': (self.vid_file.name, self.vid_file.getvalue(), 'image/jpeg'),
                    }
                    data = {
                        'conf': f'{self.conf}'
                    }
                    response = requests.post(url,files=files,data=data,stream=True)
                    annotated_result = io.BytesIO(response.content)

                    self.ann_frame.image(annotated_result)
                else:
                    url = "http://localhost:8000/api/image/cars/detect/annotated"
                    files = {
                        'image': (self.vid_file.name, self.vid_file.getvalue(), 'image/jpeg'),
                    }
                    data = {
                        'conf': f'{self.conf}'
                    }
                    response = requests.post(url,files=files,data=data,stream=True)
                    annotated_result = io.BytesIO(response.content)

                    self.ann_frame.image(annotated_result)
            elif self.source == "Video":
                if self.selected_classes[0] == 'License_Plate':
                    import websocket
                    url = "http://localhost:8000/api/video/upload"
                    files = {
                        'file': (self.vid_file.name, self.vid_file.getvalue(), 'video/mp4')
                    }
                    data = {
                        'conf': f'{self.conf}'
                    }
                    response = requests.post(url,files=files)
                    response_json = response.json()
                    sess_id = response_json['session_id']
                    video_path = response_json['video_path']
                    
                    ws = websocket.WebSocketApp(
                        f"ws://localhost:8000/api/video/ws/car/{sess_id}",
                        on_message=lambda ws, msg: self.display_frame(msg),
                        on_error=lambda ws, err: self.st.error(f"Error: {err}"),
                        on_close=lambda ws: self.st.info("Processing complete")
                    )

                    # Send video path to backend
                    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path}))
                    ws.run_forever()
                
                    print("get video stream")
                else:
                    import websocket
                    url = "http://localhost:8000/api/video/upload"
                    files = {
                        'file': (self.vid_file.name, self.vid_file.getvalue(), 'video/mp4')
                    }
                    data = {
                        'conf': f'{self.conf}'
                    }
                    response = requests.post(url,files=files)
                    response_json = response.json()
                    sess_id = response_json['session_id']
                    video_path = response_json['video_path']
                    
                    ws = websocket.WebSocketApp(
                        f"ws://localhost:8000/api/video/ws/car/{sess_id}",
                        on_message=lambda ws, msg: self.display_frame(msg),
                        on_error=lambda ws, err: self.st.error(f"Error: {err}"),
                        on_close=lambda ws: self.st.info("Processing complete")
                    )

                    # Send video path to backend
                    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path}))
                    ws.run_forever()
                
                    print("get video stream")


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()