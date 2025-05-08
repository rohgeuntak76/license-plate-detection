import os
from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi import WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import InvalidState
import uuid
import asyncio

import numpy as np
import cv2 as cv
from PIL import Image
import io
import uuid
from utils.detector import read_license_plate
from utils.license_format import write_csv

import easyocr
from ultralytics import YOLO
from utils.detector import car_detect_bytes,license_detect_bytes

car_detector = YOLO("./models/yolo11s.pt")
license_detector = YOLO("./models/yolo11s_20epochs_best.pt")
plate_reader = easyocr.Reader(['en'],gpu=False)

router = APIRouter(
    prefix="/api/video",
)   

@router.post("/plate_number/crop/detect/csv",tags=["Plate Number Detection"])
async def video_plate_number_detect(file: UploadFile,conf: float = Form(0.25)):
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    # input = file.read()
    with open(temp_path,"wb") as buffer:
        # buffer.write(input)
        buffer.write(await file.read())
    
    video = cv.VideoCapture(temp_path)
    fps = video.get(cv.CAP_PROP_FPS)
    print(fps)
    frame_nmr = -1
    vehicles = [2,3,5]
    CAR_SCORE_THRESHOLD = 0.5
    LICENSE_SCORE_THRESHOLD = 0.5
    
    results = {}
    try:
        while video.isOpened():
            frame_nmr += 1
            success, frame = video.read()
            print(frame_nmr)
            # if success:
            if success:
                results[frame_nmr] = {}

                car_results = car_detector.track(frame, persist=True,conf=CAR_SCORE_THRESHOLD)[0]
                for car_result in car_results.boxes.data.tolist():
                    x1, y1, x2, y2, track_id, score, class_id = car_result
                    if int(class_id) in vehicles:
                        vehicle_bounding_boxes = []
                        vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                        for bbox in vehicle_bounding_boxes:
                            roi = frame[int(y1):int(y2), int(x1):int(x2)] # crop the vehicle
                            # license plate detector for region of interest
                            license_plates = license_detector(roi,conf=LICENSE_SCORE_THRESHOLD)[0]
        
                            # check every bounding box for a license plate
                            for license_plate in license_plates.boxes.data.tolist():
                                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                                # crop license plate
                                plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                                # de-colorize
                                plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                                # posterize
                                _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                                ocr_detections = plate_reader.readtext(plate_treshold)

                                # OCR
                                lic_text, lic_score = read_license_plate(ocr_detections)
                                # if plate could be read write results
                                # if lic_text is not None:

                                results[frame_nmr][track_id] = {
                                    'car': {
                                        'bbox': [x1, y1, x2, y2],
                                        'bbox_score': score
                                    },
                                    'license_plate': {
                                        'bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                                        'bbox_score': plate_score,
                                        'number': lic_text,
                                        'text_score': lic_score
                                    }
                                }
            else:
                break
    finally:
        write_csv(results, './temp_video.csv')
        video.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
                        


@router.post("/upload")
async def upload_video(file: UploadFile):
    # Save uploaded video temporarily
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Return session ID for WebSocket connection
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "video_path": temp_path}

@router.websocket("/ws/car/{session_id}")
async def process_video_ws_car(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Get video path from session data (simplified here)
    data = await websocket.receive_json()
    video_path = data["video_path"]
    conf = data["conf"]
    # Process video
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_num = -1

    try:
        while cap.isOpened():
            frame_num += 1
            ret, frame = cap.read()
            ### limit 10 frame for test purpose
            if not ret or frame_num > 10:
            # if not ret :
                break
            
            return_bytes = car_detect_bytes(frame,conf,frame=True)
            # # Send bytes 
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_bytes(return_bytes)
    
            # Control processing rate to not overwhelm the connection
            await asyncio.sleep(1/fps)
    except (InvalidState,WebSocketDisconnect) as e:
        print(f"{e}")
    finally:
        print(websocket.client_state)
        print(websocket.application_state)
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            print("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)

@router.websocket("/ws/license_plate/{session_id}")
async def process_video_ws_license_plate(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Get video path from session data (simplified here)
    data = await websocket.receive_json()
    video_path = data["video_path"]
    
    # Process video
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_num = -1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            ### limit 10 frame for test purpose
            if not ret or frame_num > 10:
            # if not ret :
                break
            
            frame_num += 1
            # Process frame - detect license plates
            processed_frame = license_detector(frame)
            im_bgr = processed_frame[0].plot()
            im_rgb = im_bgr[...,::-1]
            return_image = Image.fromarray(im_rgb)
            return_bytes = io.BytesIO()
            return_image.save(return_bytes, format='JPEG', quality=95)
            return_bytes.seek(0)
            # # Convert frame to bytes and send
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_bytes(return_bytes)
    
            # Control processing rate to not overwhelm the connection
            await asyncio.sleep(1/fps)
    except (InvalidState,WebSocketDisconnect) as e:
        print(f"{e}")
    finally:
        print(websocket.client_state)
        print(websocket.application_state)
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            print("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)