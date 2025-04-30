from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2 as cv
from glob import glob
from fastapi import UploadFile,File, WebSocket,WebSocketDisconnect,websockets
from websockets.exceptions import ConnectionClosed,ConnectionClosedError,ConnectionClosedOK,InvalidState
import uuid
from ultralytics import YOLO
import asyncio
import os
from PIL import Image
import io
car_detector = YOLO("../../models/yolo11s.pt")

app = FastAPI()

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Return session ID for WebSocket connection
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "video_path": temp_path}

@app.websocket("/ws/car/{session_id}")
async def process_video_ws(websocket: WebSocket, session_id: str):
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
            if not ret or frame_num > 10:
            # if not ret :
                break
            
            frame_num += 1
            # Process frame - detect license plates
            processed_frame = car_detector(frame)
            im_bgr = processed_frame[0].plot()
            im_rgb = im_bgr[...,::-1]
            return_image = Image.fromarray(im_rgb)
            return_bytes = io.BytesIO()
            return_image.save(return_bytes, format='JPEG', quality=95)
            return_bytes.seek(0)
            # # Convert frame to bytes and send
            # _, buffer = cv.imencode('.jpg', processed_frame)
            # await websocket.send_bytes(buffer.tobytes())
            if websocket.application_state != websockets.WebSocketState.DISCONNECTED:
                await websocket.send_bytes(return_bytes)
            
            # Control processing rate to not overwhelm the connection
            await asyncio.sleep(1/fps)
    # except (WebSocketDisconnect,ConnectionClosedError):
    # except (InvalidState,ConnectionClosedOK,WebSocketDisconnect) as e:
    except (InvalidState,WebSocketDisconnect) as e:
        print(f"{e}")
        # await websocket.close(reason="Client closure")
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
# def get_stream_video():
#     # camera 정의
#     videos = glob('../../inputs/sample.mp4')
#     print(videos)
#     video = cv.VideoCapture(videos[0])
#     # cam = cv2.VideoCapture(0)
#     frame_num = -1
#     while video.isOpened():
#         # 카메라 값 불러오기
#         success, frame = video.read()

#         if not success:
#             break
#         elif frame_num < 100:
#             frame_num = frame_num + 1
#             ret, buffer = cv.imencode('.jpg', frame)
#             # frame을 byte로 변경 후 특정 식??으로 변환 후에
#             # yield로 하나씩 넘겨준다.
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
#                bytearray(frame) + b'\r\n')
            
# @app.get("/video")
# def video_streaming():
#     return StreamingResponse(get_stream_video(),media_type="multipart/x-mixed-replace; boundary=frame")
