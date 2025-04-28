from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse

from utils.detector import car_detect_bytes,license_detect_bytes


router = APIRouter(
    prefix="/api/image",
)

@router.post("/cars/detect/annotated" , tags=["Cars Detection"])
def image_car_detect(image: UploadFile,conf: float = Form(0.25)):
    """
    Get Car Image, Return Annotated Image with Car detection by Yolo
    
    Args:
        image : Car image
        conf : confidence threshold
    Returns:
        Image (BytesIO): Annotated Image with Car detection by Yolo
    """
    return_bytes = car_detect_bytes(image,conf)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post("/plates/detect/annotated", tags=["Plates Detection"])
def image_license_plate_detect(image: UploadFile,conf: float = Form(0.25)):
    """
    Get Car Image , Return Annotated Image with license detection by Yolo ( No cropping )
    
    Args:
        image : Car Image
        conf : Confidence Threshold of Yolo model
    Returns:
        Image (BytesIO): Annotated Image with license detection by Yolo
    """
    return_bytes = license_detect_bytes(image,conf)

    return StreamingResponse(content=return_bytes,media_type="image/jpeg")
