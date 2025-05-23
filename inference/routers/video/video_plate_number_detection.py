import os
from fastapi import APIRouter
from fastapi import WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import InvalidState
import asyncio

import cv2 as cv

from utils.detector import crop_vehicle_license_then_read,reset_tracker


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
            detection_results = crop_vehicle_license_then_read(input_image=frame,vehicle_conf=vehicle_conf,license_conf=license_conf,frame_number=frame_num)
            # # Convert frame to bytes and send
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_json(detection_results)
    
            # Control processing rate to not overwhelm the connection
            await asyncio.sleep(1/fps)
    except (InvalidState,WebSocketDisconnect) as e:
        print(f"{e}")
    finally:
        print(websocket.client_state)
        print(websocket.application_state)
        if reset_tracker():
            print('tracker reset done!')
        if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close(reason="Normal closure")
        else:
            print("Connection is already closed!")
        cap.release()
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)