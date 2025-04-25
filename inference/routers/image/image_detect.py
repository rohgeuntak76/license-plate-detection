from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io

from utils import get_bytes_from_prediction

car_detector = YOLO("./models/yolo11s.pt")
license_detector = YOLO("./models/yolo11s_20epochs_best.pt")


router = APIRouter(
    prefix="/api/image",
)

@router.post("/car_detect")
def image_car_detect(image: UploadFile,conf: float = Form(0.25)):
    file = image.file.read()
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    prediction = car_detector(input_image,conf=conf,classes=[2,5,7])
    return_bytes = get_bytes_from_prediction(prediction,quality=95)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post("/license_plate_detect")
def image_license_plate_detect(image: UploadFile,conf: float = Form(0.25)):
    file = image.file.read()
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    prediction = license_detector(input_image,conf=conf)
    return_bytes = get_bytes_from_prediction(prediction,quality=95)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")