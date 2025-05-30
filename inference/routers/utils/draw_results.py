from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi import WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import InvalidState

import json
import numpy as np
import pandas as pd
import cv2 as cv
# from utils.detector import draw_border, get_bytes_from_prediction
from utils.detector import get_bytes_from_prediction
import asyncio
import os


router = APIRouter(
    prefix="/api/utils",
)

@router.post("/draw/annotated", tags=["Utils"])
def image_info_ann(image: UploadFile,item_json: str = Form()):
    """
    Get Image and detection info -> Draw and Return Annotated Image
    
    Args:
        image : vehicle Image
        info : Detection Info
        
    Returns:
        Image (BytesIO): Annotated Image with license number detection by Yolo
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    info = json.loads(item_json)
    
    for object in range(len(info)):
        vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = info[object]['vehicle_bbox']
        # draw_border(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 12, line_length_x=200, line_length_y=200)
        cv.rectangle(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 8)
        plate_x1, plate_y1, plate_x2, plate_y2 = info[object]['lp_bbox']
        roi = input_image[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
        cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

        # write detected number
        (text_width, text_height), _ = cv.getTextSize(info[object]["lp_number"],cv.FONT_HERSHEY_DUPLEX,2,6)
        cv.putText(
                roi,
                info[object]["lp_number"],
                (int((plate_x2 + plate_x1 - text_width)/2), int(plate_y1 - text_height)),
                cv.FONT_HERSHEY_DUPLEX,
                2,
                (0, 255, 0),
                6
            )   
    
    return_bytes = get_bytes_from_prediction(input_image,quality=95)
    # return_bytes.seek(0)
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")


@router.websocket("/ws/draw/annotated/{session_id}")
async def video_info_ann(websocket: WebSocket):
    """
    Get video and detection info -> Draw and Return Annotated Image
    
    Args:
        Video : Traffic Video
        info : Detection Info
        
    Returns:
        Image (BytesIO): Annotated Image with license number detection by Yolo
    """
    await websocket.accept()

    data = await websocket.receive_json()
    video_path = data["video_path"]
    detection_results = data["detection_results"]
    results_df = pd.DataFrame(detection_results)
    # print(results_df[:10])
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
            df_ = results_df[results_df['frame_number'] == frame_num]
            for index in range(len(df_)):
                # draw vehicle
                # if str(df_.iloc[index]['lp_number']) != 'nan':
                    print(df_.iloc[index]['lp_number'])
                    vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = df_.iloc[index]['vehicle_bbox']
                    
                    # draw_border(frame, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0),12, line_length_x=200, line_length_y=200)
                    cv.rectangle(frame, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 8)
                    # draw license plate
                    plate_x1, plate_y1, plate_x2, plate_y2 = df_.iloc[index]['lp_bbox']

                    # region of interest
                    roi = frame[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
                    cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

                    # write detected number
                    # (text_width, text_height), _ = cv.getTextSize(str(df_.iloc[index]['lp_number']),cv.FONT_HERSHEY_SIMPLEX,2,6)

                    # cv.putText(frame,str(df_.iloc[index]['lp_number']),(int((vhcl_x2 + vhcl_x1 - text_width)/2), int(vhcl_y1 - text_height)),cv.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0),6)
                    
                    (text_width, text_height), _ = cv.getTextSize(str(df_.iloc[index]['lp_number']),cv.FONT_HERSHEY_DUPLEX,2,6)
                    cv.putText(
                            roi,
                            str(df_.iloc[index]['lp_number']),
                            (int((plate_x2 + plate_x1 - text_width)/2), int(plate_y1 - text_height)),
                            cv.FONT_HERSHEY_DUPLEX,
                            2,
                            (0, 255, 0),
                            6
                        )
                    return_bytes = get_bytes_from_prediction(frame,quality=95)
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