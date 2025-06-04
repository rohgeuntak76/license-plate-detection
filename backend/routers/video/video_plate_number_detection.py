import os
from fastapi import APIRouter
from fastapi import WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import InvalidState
import asyncio

import cv2 as cv

from utils.detector import crop_vehicle_license_then_read,reset_tracker
from utils.logging import logger

router = APIRouter(
    prefix="/api/video",
)   

@router.websocket("/ws/license_number/{session_id}")
async def process_video_ws_license_number(websocket: WebSocket, session_id: str):
    '''
    Does not Use Tracker
    return info ( not image )
    '''
    await websocket.accept()
    
    # Get video path from session data (simplified here)
    data = await websocket.receive_json()
    video_path = data["video_path"]
    vehicle_conf = data["vehicle_conf"]
    license_conf = data["license_conf"]
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
            detection_results = crop_vehicle_license_then_read(input_image=frame,vehicle_conf=vehicle_conf,license_conf=license_conf,frame_number=frame_num)
            # Convert frame to bytes and send
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_json(detection_results)
    
            # # Control processing rate to not overwhelm the connection ( without this sleep, dataframe visualize and interrupt inference won't work)
            await asyncio.sleep(1/fps)

    except (InvalidState,WebSocketDisconnect) as e:
        logger.error(f"{e}")
    finally:
        logger.info(f"websocket client state : {websocket.client_state}. websocket application state : {websocket.application_state}")
        logger.info(f"Inferenced # of Frame : {frame_num} / {total_frame}")
        if reset_tracker():
            logger.info('tracker reset done!')
        
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            logger.info("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)