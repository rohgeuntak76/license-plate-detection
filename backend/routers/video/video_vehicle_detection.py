import os
from fastapi import APIRouter
from fastapi import WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import InvalidState
import asyncio

import cv2 as cv

from utils.detector import vehicle_detect_bytes,license_detect_bytes
from utils.logging import logger

router = APIRouter(
    prefix="/api/video",
)   

@router.websocket("/ws/vehicle/{session_id}")
async def process_video_ws_vehicle(websocket: WebSocket, session_id: str):
    '''
    Does not Use Tracker
    '''
    await websocket.accept()
    
    # Get video path from session data (simplified here)
    data = await websocket.receive_json()
    video_path = data["video_path"]
    conf = data["conf"]
    classes = data["classes"]
    ratio = data["ratio"]

    # Process video
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    thres_frame = total_frame*ratio
    frame_num = -1

    try:
        while cap.isOpened():
            frame_num += 1
            ret, frame = cap.read()
            ### limit 10 frame for test purpose
            if not ret or frame_num > thres_frame:
            # if not ret :
                break

            return_bytes = vehicle_detect_bytes(frame,conf,classes)
            # Send bytes 
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_bytes(return_bytes)
    
            # # Control processing rate to not overwhelm the connection
            # await asyncio.sleep(1/fps)
            await asyncio.sleep(0.1)
    except (InvalidState,WebSocketDisconnect) as e:
        logger.error(f"{e}")
    finally:
        logger.info(f"websocket client state : {websocket.client_state}. websocket application state : {websocket.application_state}")
        logger.info(f"Inferenced # of Frame : {frame_num} / {total_frame}")
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            logger.info("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)

@router.websocket("/ws/license_plate/{session_id}")
async def process_video_ws_license_plate(websocket: WebSocket, session_id: str):
    '''
    Does not Use Tracker
    '''
    await websocket.accept()
    
    # Get video path from session data (simplified here)
    data = await websocket.receive_json()
    video_path = data["video_path"]
    conf = data["conf"]
    ratio = data["ratio"]

    # Process video
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    thres_frame = total_frame*ratio
    frame_num = -1

    try:
        while cap.isOpened():
            frame_num += 1
            ret, frame = cap.read()
            ### limit 10 frame for test purpose
            if not ret or frame_num > thres_frame:
            # if not ret :
                break
            
            # Process frame - detect license plates
            return_bytes = license_detect_bytes(frame,conf)
            # Convert frame to bytes and send
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_bytes(return_bytes)
    
            # # Control processing rate to not overwhelm the connection
            # await asyncio.sleep(1/fps)
            await asyncio.sleep(0.1)
    except (InvalidState,WebSocketDisconnect) as e:
        logger.error(f"{e}")
    finally:
        logger.info(f"websocket client state : {websocket.client_state}. websocket application state : {websocket.application_state}")
        logger.info(f"Inferenced # of Frame : {frame_num} / {total_frame}")
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            logger.info("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)