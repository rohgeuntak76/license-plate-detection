from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

import cv2 as cv
import numpy as np

from utils.detector import crop_vehicle_license_then_read, license_detect_number, reset_tracker , get_bytes_from_prediction
from utils.logging import logger

class DetectionResult(BaseModel):
    """Single vehicle and license plate detection result"""
    frame_number: int = Field(..., description="Frame number in sequence")
    track_id: int = Field(..., description="Unique tracking ID for the vehicle")
    vehicle_bbox: list[float] = Field(..., description="Vehicle bounding box [x1, y1, x2, y2]")
    vehicle_bbox_score: float = Field(..., ge=0.0, le=1.0, description="Vehicle detection confidence score")
    lp_bbox: list[float] = Field(..., description="License plate bounding box [x1, y1, x2, y2]")
    lp_bbox_score: float = Field(..., ge=0.0, le=1.0, description="License plate detection confidence score")
    lp_number: Optional[str] = Field(default=None, description="Detected license plate text")
    lp_text_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="OCR confidence score")
    class Config:
        json_schema_extra = {
            "example": {
                "frame_number": 1,
                "track_id": 123,
                "vehicle_bbox": [100.5, 200.3, 350.7, 450.2],
                "vehicle_bbox_score": 0.95,
                "lp_bbox": [150.1, 220.4, 280.8, 250.6],
                "lp_bbox_score": 0.88,
                "lp_number": "ABC123",
                "lp_text_score": 0.92
            }
        }

router = APIRouter(
    prefix="/api/image/plate_number",
)

@router.post(
    "/crop/detect/info",
    tags=["Plate Number Detection"],
    response_model=list[DetectionResult],
    summary="Detect license Numbers in Image",
    description="Get vehicle Image -> Crop vehicle -> Crop license plate -> Read Plate Number ( OCR ) -> Return Dict",
)
def image_license_plate_number_crop_info(image: UploadFile,vehicle_conf: float = Form(0.25),license_conf: float = Form(0.25)):
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    results = crop_vehicle_license_then_read(input_image,vehicle_conf,license_conf)
    if reset_tracker():
        logger.info('tracker reset done!')
    return results

@router.post(
    "/crop/detect/annotated",
    tags=["Plate Number Detection"],
    response_class=StreamingResponse,
    summary="Detect license Numbers in Image",
    description="Get vehicle Image -> Crop vehicle -> Crop license plate -> Read Plate Number ( OCR ) -> Return Annotated Image",
)
def image_license_plate_number_crop_ann(image: UploadFile,vehicle_conf: float = Form(0.25),license_conf: float = Form(0.25)):
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    results = crop_vehicle_license_then_read(input_image,vehicle_conf=vehicle_conf,license_conf=license_conf)
    if reset_tracker():
        logger.info('tracker reset done!')
    
    for object in range(len(results)):
        # Draw Vehicle Bbox
        vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = results[object]['vehicle_bbox']
        cv.rectangle(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 8)
        
        # Draw License Plate Bbox
        plate_x1, plate_y1, plate_x2, plate_y2 = results[object]['lp_bbox']
        roi = input_image[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
        cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

        # write detected number
        (text_width, text_height), _ = cv.getTextSize(results[object]["lp_number"],cv.FONT_HERSHEY_DUPLEX,2,6)
        cv.putText(
                roi,
                results[object]["lp_number"],
                (int((plate_x2 + plate_x1 - text_width)/2), int(plate_y1 - text_height)),
                cv.FONT_HERSHEY_DUPLEX,
                2,
                (0, 255, 0),
                6
            )
    return_bytes = get_bytes_from_prediction(input_image,quality=95)
    
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post(
    "/detect/info",
    tags=["Experimental"],
    # response_class=StreamingResponse,
    summary="Detect license Numbers in Cropped Image",
    description="Get Cropped license Plate Image , Return Plate number and plate score",
)
def image_license_plate_number_info(image: UploadFile):
    lic_text, lic_score = license_detect_number(image)

    return {"plate_number": lic_text, "plate_score": lic_score}

