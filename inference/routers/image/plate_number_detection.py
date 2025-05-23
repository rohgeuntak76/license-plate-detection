from fastapi import APIRouter
from fastapi import UploadFile, Form
from fastapi.responses import StreamingResponse
import cv2 as cv
import numpy as np

from utils.detector import draw_border, crop_vehicle_license_then_read, license_detect_number, reset_tracker , get_bytes_from_prediction

router = APIRouter(
    prefix="/api/image/plate_number",
)

@router.post("/crop/detect/info", tags=["Plate Number Detection"])
def image_license_plate_number_crop_info(image: UploadFile,vehicle_conf: float = Form(0.25),license_conf: float = Form(0.25)):
    """
    Get vehicle Image -> Crop vehicle -> Crop license plate -> Read Plate Number ( OCR ) -> Return Dict
    
    Args:
        image (UploadFile): vehicle Image
    
    Returns:
        list of Dict : [
                    {
                        'frame_number': frame_number,
                        'track_id': track_id,
                        'vehicle_bbox': [x1, y1, x2, y2],
                        'vehicle_bbox_score': score,
                        'lp_bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                        'lp_bbox_score': plate_score,
                        'lp_number': lic_text,
                        'lp_text_score': lic_score
                    },
        ]
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    results = crop_vehicle_license_then_read(input_image,vehicle_conf,license_conf)
    if reset_tracker():
        print('tracker reset done!')
    return results

@router.post("/crop/detect/annotated", tags=["Plate Number Detection"])
def image_license_plate_number_crop_ann(image: UploadFile,vehicle_conf: float = Form(0.25),license_conf: float = Form(0.25)):
    """
    Get vehicle Image -> Crop vehicle -> Crop License plate -> Read Plate Number ( OCR ) -> Return Annotated Image
    
    Args:
        image : vehicle Image
        
    Returns:
        Image (BytesIO): Annotated Image with license number detection by Yolo
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 
    

    results = crop_vehicle_license_then_read(input_image,vehicle_conf=vehicle_conf,license_conf=license_conf)
    if reset_tracker():
        print('tracker reset done!')
    
    for object in range(len(results)):
        vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = results[object]['vehicle_bbox']
        draw_border(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 12, line_length_x=200, line_length_y=200)
        
        plate_x1, plate_y1, plate_x2, plate_y2 = results[object]['lp_bbox']
        roi = input_image[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
        cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

        # write detected number
        (text_width, text_height), _ = cv.getTextSize(results[object]["lp_number"],cv.FONT_HERSHEY_SIMPLEX,2,6)
        cv.putText(
                input_image,
                results[object]["lp_number"],
                (int((vhcl_x2 + vhcl_x1 - text_width)/2), int(vhcl_y1 - text_height)),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                6
            )
    return_bytes = get_bytes_from_prediction(input_image,quality=95)
    
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post("/detect/info", tags=["Experimental"])
def image_license_plate_number_info(image: UploadFile):
    """
    Get Cropped license Plate Image , Return Plate number and plate score
    
    Args:
        image (UploadFile): Cropped license Plate Image
    
    Returns:
        Dict : Plate number and plate score
    """
    lic_text, lic_score = license_detect_number(image)

    return {"plate_number": lic_text, "plate_score": lic_score}

