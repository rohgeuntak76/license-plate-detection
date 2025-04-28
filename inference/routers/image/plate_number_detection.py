from fastapi import APIRouter
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import cv2 as cv
import numpy as np
import io

from utils import draw_border, crop_car_license_then_read, license_detect_number

router = APIRouter(
    prefix="/api/image/plate_number",
)

@router.post("/detect/info", tags=["Plate Number Detection"])
def image_license_plate_detect(image: UploadFile):
    """
    Get Cropped license Plate Image , Return Plate number and plate score
    
    Args:
        image (UploadFile): Cropped license Plate Image
    
    Returns:
        Dict : Plate number and plate score
    """
    lic_text, lic_score = license_detect_number(image)

    return {"plate_number": lic_text, "plate_score": lic_score}


@router.post("/crop/detect/info", tags=["Plate Number Detection"])
def image_license_plate_detect(image: UploadFile):
    """
    Get Car Image -> Crop car -> Crop license plate -> Read Plate Number ( OCR ) -> Return Dict
    
    Args:
        image (UploadFile): Car Image
    
    Returns:
        Dict : 'car': {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_score': score
                },
                'license_plate': {
                    'bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                    'bbox_score': plate_score,
                    'number': lic_text,
                    'text_score': lic_score
                }
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 

    results = crop_car_license_then_read(input_image)

    return results

@router.post("/crop/detect/annotated", tags=["Plate Number Detection"])
def image_license_plate_detect(image: UploadFile):
    """
    Get Car Image -> Crop Car -> Crop License plate -> Read Plate Number ( OCR ) -> Return Annotated Image
    
    Args:
        image : Car Image
        
    Returns:
        Image (BytesIO): Annotated Image with license number detection by Yolo
    """
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 
    
    results = crop_car_license_then_read(input_image)
 
    for track_id in results[0].keys():
        vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = results[0][track_id]['car']['bbox']
        draw_border(input_image, (int(vhcl_x1), int(vhcl_y1)),(int(vhcl_x2), int(vhcl_y2)), (0, 255, 0), 12, line_length_x=200, line_length_y=200)
        
        plate_x1, plate_y1, plate_x2, plate_y2 = results[0][track_id]['license_plate']['bbox']
        roi = input_image[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
        cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

        # write detected number
        (text_width, text_height), _ = cv.getTextSize(results[0][track_id]["license_plate"]["number"],cv.FONT_HERSHEY_SIMPLEX,2,6)
        cv.putText(
                input_image,
                results[0][track_id]["license_plate"]["number"],
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


