# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

import requests
import json
import pandas as pd

api_host = "localhost:8000"
# logo_url = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
logo_url = "assets/hpe_pri_grn_pos_rgb.png"
detection_models = ["Yolo11s","Yolo11s_20epochs_best"]

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
        self.vehicle_conf = 0.25
        self.license_conf = 0.25
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file = None
        self.selected_ind = []  # List of selected class indices for detection
        self.selected_classes = []
        self.model = None  # YOLO model instance
        self.usecase = None
        self.api_host = api_host
        self.temp_dict = {"model": None, **kwargs}
        self.model_dir = None
        self.model_path = None  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]
        
        if 'stage' not in self.st.session_state:
            self.st.session_state.stage = 0

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#01A982; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">License Plate Number Detection Application</h1></div>"""

        # Subtitle of streamlit application
        # sub_title_cfg = """<div><h4 style="color:#000000; text-align:center; font-family: 'Archivo', sans-serif; 
        # margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        # of Ultralytics YOLO! 🚀</h4></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="Licenses Number Detection App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        # self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def usecase_sidebar(self):
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = logo_url
            self.st.image(logo, width=250)

        self.st.sidebar.title("Use Case")  # Add elements to vertical setting menu
        self.usecase = self.st.sidebar.selectbox(
            "Select Use Case",
            ("Vehicle Detection", "License Number Detection"),
            on_change=self.set_state,args=[0]
        )
    
    def license_sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        self.st.sidebar.title("Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Data Type",
            ("Image", "Video"),
            on_change=self.set_state,args=[0]
        )  # Add source selection dropdown
        
        self.vehicle_conf = float(
            self.st.sidebar.slider("vehicle Confidence Threshold", 0.0, 1.0, self.vehicle_conf, 0.01,on_change=self.set_state,args=[0])
        )
        self.license_conf = float(
            self.st.sidebar.slider("License Confidence Threshold", 0.0, 1.0, self.license_conf, 0.01,on_change=self.set_state,args=[0])
        )  # Slider for confidence
        # self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame

    def vehicle_sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        self.st.sidebar.title("Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Data Type",
            ("Image", "Video"),
            on_change=self.set_state,args=[0]
        )  # Add source selection dropdown
        
        self.vehicle_conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.vehicle_conf, 0.01,on_change=self.set_state,args=[0])
        )  # Slider for confidence
        # self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        
        if self.vid_file is None:
            if self.source == "Video":
                self.vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"],on_change=self.set_state,args=[0])
                if self.vid_file is not None:
                    self.org_frame.video(self.vid_file,muted=True)    
                    
            elif self.source == "Image":
                self.vid_file = self.st.sidebar.file_uploader("Upload Image File", type=["jpeg", "png"],on_change=self.set_state,args=[0])
                if self.vid_file is not None:
                    self.org_frame.image(self.vid_file)

    def license_configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection
        vehicle_detectors = ["Yolo11s"]
        license_detectors = ["Yolo11s_20epochs_best"]
        vehicle_selected_model = self.st.sidebar.selectbox("vehicle Detector", vehicle_detectors)
        license_selected_model = self.st.sidebar.selectbox("License Plate Detector", license_detectors)

    def vehicle_configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection
        available_models = detection_models
        
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is loading..."):
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
        image = io.BytesIO(frame_bytes)
        # Update the placeholder
        self.ann_frame.image(image)

    def display_dataframe(self,detection_results,results_list,result_df):
        # update list
        results_json = json.loads(detection_results)
        results_list.extend(results_json)
        result_df.add_rows(results_json)

    def set_state(self,i):
        '''
        stage 0 : intial stage
        stage 1 : Do inference and Get the Inference results
        stage 2 : Visualize the outputs
        '''
        self.st.session_state.stage = i

    def inference(self):
        """Perform vehicle detection or license plate number inference on video or image."""
        self.web_ui()  # Initialize the web interface
        self.usecase_sidebar()
        if self.usecase == "Vehicle Detection" :
            self.vehicle_sidebar()  # Create the sidebar
            self.source_upload()  # Upload the video source
            self.vehicle_configure()  # Configure the app
            side_left, side_right = self.st.sidebar.columns(2)

            if side_left.button("Start",use_container_width=True):
                self.model_load.empty()   
                side_right.button("Clear",use_container_width=True)  # Button to stop the inference
                if self.source == "Image":
                    LOGGER.info(f"Selected Classes : {self.selected_classes}")
                    if self.selected_classes[0] == 'License_Plate': # license plate detection usecase
                        url = "http://" + api_host + "/api/image/plates/detect/annotated"
                    else: # vehicle detction usecase
                        url = "http://" + api_host + "/api/image/vehicles/detect/annotated"
                    # Do inference
                    files = {
                            'image': (self.vid_file.name, self.vid_file.getvalue(), 'image/jpeg'),
                        }
                    data = {
                        'conf': f'{self.vehicle_conf}'
                    }
                    response = requests.post(url,files=files,data=data,stream=True)
                    annotated_result = io.BytesIO(response.content)

                    self.ann_frame.image(annotated_result)
                elif self.source == "Video":
                    import websocket
                    if self.selected_classes[0] == 'License_Plate':
                        ws_url = f"ws://{api_host}/api/video/ws/license_plate/"
                    else:
                        ws_url = f"ws://{self.api_host}/api/video/ws/vehicle/"

                    # Upload Video
                    upload_url = "http://" + api_host + "/api/utils/video/upload"
                    files = {
                        'file': (self.vid_file.name, self.vid_file.getvalue(), 'video/mp4')
                    }
                    data = {
                        'conf': f'{self.vehicle_conf}'
                    }
                    response = requests.post(upload_url,files=files)
                    response_json = response.json()
                    sess_id = response_json['session_id']
                    video_path = response_json['video_path']
                    # Do inference
                    ws_url = ws_url + sess_id
                    ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=lambda ws, msg: self.display_frame(msg),
                        on_error=lambda ws, err: self.st.error(f"Error: {err}"),
                        on_close=lambda ws,status,msg: self.st.info(f"Processing complete. code {status} : {msg}")
                    )

                    # Send video path to backend
                    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"conf":self.vehicle_conf}))
                    ws.run_forever()
                    
                        
        elif self.usecase == 'License Number Detection':
            self.license_sidebar()
            self.source_upload()
            self.license_configure()
            side_left, side_right = self.st.sidebar.columns(2)
            side_left.button("Start",on_click=self.set_state, args=[1],use_container_width=True)
            side_right.button("Clear",on_click=self.set_state, args=[0],use_container_width=True)

            if self.st.session_state.stage == 0:
                self.st.session_state.detection_result = None
                LOGGER.info(f"Stage 0 : {self.st.session_state}")
            
            if self.st.session_state.stage == 1:
                if self.source == "Image":
                    files = {
                        'image': (self.vid_file.name, self.vid_file.getvalue(), 'image/jpeg'),
                    }
                    data = {
                        'vehicle_conf': f'{self.vehicle_conf}',
                        'license_conf': f'{self.license_conf}',
                    }
                    
                    info_url = url = "http://" + api_host + "/api/image/plate_number/crop/detect/info"
                    response_info = requests.post(info_url,files=files,data=data)
                    response_info_json = response_info.json()
                    LOGGER.info(response_info_json)
                    self.st.session_state.detection_result = response_info_json
                    self.st.dataframe(response_info_json)
                    self.st.button("Visualize",on_click=self.set_state, args=[2])
                elif self.source == "Video":
                    import websocket
                    url = "http://" + api_host + "/api/utils/video/upload"
                    files = {
                        'file': (self.vid_file.name, self.vid_file.getvalue(), 'video/mp4')
                    }
                    
                    response = requests.post(url,files=files)
                    response_json = response.json()
                    sess_id = response_json['session_id']
                    video_path = response_json['video_path']
                    results_list = []
                    result_df = self.st.dataframe(pd.DataFrame(columns=("frame_number","track_id","vehicle_bbox","vehicle_bbox_score","lp_bbox","lp_bbox_score","lp_number","lp_text_score")),hide_index=True)
                    ws = websocket.WebSocketApp(
                        f"ws://{api_host}/api/video/ws/license_number/{sess_id}",
                        on_message=lambda ws, msg: self.display_dataframe(msg,results_list,result_df),
                        on_error=lambda ws, err: self.st.error(f"Error: {err}"),
                        on_close=lambda ws,status,msg: self.st.info(f"Processing complete. code {status} : {msg}")
                    )
                    # Send video path to backend
                    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"vehicle_conf": self.vehicle_conf,"license_conf":self.license_conf}))
                    ws.run_forever()
              
                    self.st.session_state.detection_result = results_list
                    # self.st.dataframe(results_list)
                    self.st.button("Visualize",on_click=self.set_state, args=[2])


            if self.st.session_state.stage == 2:
                if self.source == "Image":
                    files = {
                        'image': (self.vid_file.name, self.vid_file.getvalue(), 'image/jpeg'),
                    }
                    data = {
                        'item_json': json.dumps(self.st.session_state.detection_result),
                    }

                    url = 'http://' + api_host + '/api/utils/draw/annotated'
                    response = requests.post(url,files=files,data=data,stream=True)
                    annotated_result = io.BytesIO(response.content)

                    self.ann_frame.image(annotated_result)
                    self.st.dataframe(self.st.session_state.detection_result)
                    self.st.button("Visualize",on_click=self.set_state, args=[2])
                elif self.source == "Video":
                    import websocket

                    url = "http://" + api_host + "/api/utils/video/upload"
                    files = {
                        'file': (self.vid_file.name, self.vid_file.getvalue(), 'video/mp4')
                    }
                    
                    response = requests.post(url,files=files)
                    response_json = response.json()
                    sess_id = response_json['session_id']
                    video_path = response_json['video_path']
                    
                    ws = websocket.WebSocketApp(
                        f"ws://{self.api_host}/api/utils/ws/draw/annotated/{sess_id}",
                        on_message=lambda ws,msg: self.display_frame(msg),
                        on_error=lambda ws,err: self.st.error(f"Error: {err}"),
                        on_close=lambda ws,status,msg: self.st.info(f"Processing complete. code {status} : {msg}")
                    )

                    # Send video path to backend
                    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"detection_results":self.st.session_state.detection_result}))
                    ws.run_forever()
                
                    self.st.dataframe(self.st.session_state.detection_result)
                    self.st.button("Visualize",on_click=self.set_state, args=[2])

if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()