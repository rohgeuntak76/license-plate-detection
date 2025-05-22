from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse

import uuid
import json
import numpy as np
import cv2 as cv
from utils.detector import draw_border
from PIL import Image
import io



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


@router.post("/draw/annotated", tags=["Utils"])
def image_info_ann(image: UploadFile,item_json: str = Form()):
    """
    Get Image and detection info -> Draw and Return Annotated Image
    
    Args:
        image : Car Image
        info : Detection Info
        
    Returns:
        Image (BytesIO): Annotated Image with license number detection by Yolo
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    info = json.loads(item_json)
    
    for object in range(len(info)):
        vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = info[object]['car_bbox']
        draw_border(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 12, line_length_x=200, line_length_y=200)
        
        plate_x1, plate_y1, plate_x2, plate_y2 = info[object]['lp_bbox']
        roi = input_image[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
        cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

        # write detected number
        (text_width, text_height), _ = cv.getTextSize(info[object]["lp_number"],cv.FONT_HERSHEY_SIMPLEX,2,6)
        cv.putText(
                input_image,
                info[object]["lp_number"],
                (int((vhcl_x2 + vhcl_x1 - text_width)/2), int(vhcl_y1 - text_height)),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                6
            )
    im_rgb = input_image[...,::-1]
    return_image = Image.fromarray(im_rgb)
    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=95)
    return_bytes.seek(0)
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")