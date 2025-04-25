from fastapi import APIRouter
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io

car_detector = YOLO("./models/yolo11s.pt")
license_detector = YOLO("./models/yolo11s_20epochs_best.pt")


router = APIRouter(
    prefix="/api/image",
)

@router.post("/car_detect")
def image_car_detect(file: bytes = File()):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")

    prediction = car_detector(input_image)
    # print(prediction)
    im_bgr = prediction[0].plot()
    im_rgb = im_bgr[...,::-1]
    return_image = Image.fromarray(im_rgb)
    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=95)
    return_bytes.seek(0)
    
    # print(objects)
    # return {"result": detect_result}
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")

@router.post("/license_plate_detect")
def image_license_plate_detect(file: bytes = File()):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    prediction = license_detector(input_image)
    # print(prediction)
    # return_image = prediction[0].plot(pil=True)
    im_bgr = prediction[0].plot()
    im_rgb = im_bgr[...,::-1]
    return_image = Image.fromarray(im_rgb)

    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=95)
    return_bytes.seek(0)
    
    # print(objects)
    # return {"result": detect_result}
    return StreamingResponse(content=return_bytes,media_type="image/jpeg")