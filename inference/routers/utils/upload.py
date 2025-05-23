from fastapi import APIRouter
from fastapi import UploadFile

import uuid

router = APIRouter(
    prefix="/api/utils",
)

@router.post("/video/upload",tags=['Utils'])
async def upload_video(file: UploadFile):
    # Save uploaded video temporarily
    temp_path = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Return session ID for WebSocket connection
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "video_path": temp_path}