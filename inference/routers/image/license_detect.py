from fastapi import APIRouter
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from utils import get_bytes_from_prediction

car_detector = YOLO("./models/yolo11s.pt")
license_detector = YOLO("./models/yolo11s_20epochs_best.pt")


router = APIRouter(
    prefix="/api/image",
)

# def get_bytes_from_prediction(prediction: np.ndarray,quality: int) -> bytes:
#     """
#     Convert YOLO's prediction to Bytes
    
#     Args:
#     prediction (np.ndarray): A YOLO's prediction
    
#     Returns:
#     bytes : BytesIO object that contains the image in JPEG format with quality 95
#     """
#     im_bgr = prediction[0].plot()
#     im_rgb = im_bgr[...,::-1]
#     return_image = Image.fromarray(im_rgb)
#     return_bytes = io.BytesIO()
#     return_image.save(return_bytes, format='JPEG', quality=quality)
#     return_bytes.seek(0)
#     return return_bytes

@router.post("/car_detect")
def image_car_detect(file: bytes = File(),conf: float = Form(0.25)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    prediction = car_detector(input_image,conf=conf)
    return_bytes = get_bytes_from_prediction(prediction,quality=95)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post("/license_plate_detect")
def image_license_plate_detect(file: bytes = File(),conf: float = Form(0.25)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    prediction = license_detector(input_image,conf=conf)
    return_bytes = get_bytes_from_prediction(prediction,quality=95)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")