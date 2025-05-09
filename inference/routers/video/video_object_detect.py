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

import easyocr
from ultralytics import YOLO

from utils.license_format import write_csv
from utils.detector import car_detect_bytes,license_detect_bytes, crop_car_license_then_read
# from utils.detector import read_license_plate
# from utils.detector import car_tracker_numpy,license_detect_numpy

# car_detector = YOLO("./models/yolo11s.pt")
# license_detector = YOLO("./models/yolo11s_20epochs_best.pt")
# plate_reader = easyocr.Reader(['en'],gpu=False)

router = APIRouter(
    prefix="/api/video",
)   

@router.post("/plate_number/crop/detect/info",tags=["Plate Number Detection"])
async def video_plate_number_detect(file: UploadFile,car_conf: float = Form(0.25),license_conf: float = Form(0.25)):
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_path,"wb") as buffer:
        buffer.write(await file.read())
    
    video = cv.VideoCapture(temp_path)
    fps = video.get(cv.CAP_PROP_FPS)
    print(fps)
    frame_nmr = -1
    
    results = {}
    try:
        while video.isOpened():
            frame_nmr += 1
            success, frame = video.read()
            # print(frame_nmr)
            # if success:
            if success and frame_nmr < 10:
                results[frame_nmr] = crop_car_license_then_read(frame,car_conf,license_conf)
            else:
                break
    finally:
        write_csv(results, './temp_video.csv')
        video.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
                        


@router.post("/upload",tags=['Utils'])
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
    conf = data["conf"]
    
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
            return_bytes = license_detect_bytes(frame,conf,frame=True)
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