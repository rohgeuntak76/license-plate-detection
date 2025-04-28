from PIL import Image
import io
import numpy as np
import string
import cv2 as cv
import easyocr
from ultralytics import YOLO

car_detector = YOLO("./models/yolo11s.pt")
license_detector = YOLO("./models/yolo11s_20epochs_best.pt")
plate_reader = easyocr.Reader(['en'],gpu=False)

def car_detect_bytes(image,conf):
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR)  # bytes -> numpy 

    prediction = car_detector(input_image,conf=conf,classes=[2,5,7])
    return_bytes = get_bytes_from_prediction(prediction,quality=95)
    return return_bytes

def license_detect_bytes(image,conf):
    file = image.file.read()
    image_np = np.frombuffer(file, np.uint8)
    input_image = cv.imdecode(image_np, cv.IMREAD_COLOR) 
    
    prediction = license_detector(input_image,conf=conf)
    return_bytes = get_bytes_from_prediction(prediction,quality=95)
    return return_bytes

def license_detect_number(image):
    file = image.file.read()
    # input_image = Image.open(io.BytesIO(file)).convert("RGB")
    ocr_detections = plate_reader.readtext(file)
    lic_text, lic_score = read_license_plate(ocr_detections)
    return lic_text, lic_score

def get_bytes_from_prediction(prediction: np.ndarray,quality: int) -> bytes:
    """
    Convert YOLO's prediction to Bytes
    
    Args:
    prediction (np.ndarray): A YOLO's prediction
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 95
    """
    im_bgr = prediction[0].plot()
    im_rgb = im_bgr[...,::-1]
    return_image = Image.fromarray(im_rgb)
    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=quality)
    return_bytes.seek(0)
    return return_bytes

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def license_complies_format(text):
    # True if the license plate complies with the format, False otherwise.
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_char_to_int, 
        3: dict_char_to_int,
        4: dict_int_to_char, 
        5: dict_int_to_char, 
        6: dict_int_to_char
    }
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(detections):

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return format_license(text), score

    return None, None

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def crop_car_license_then_read(input_image):
    results = {}
    results[0] = {}
    car_results = car_detector.track(input_image, persist=True,conf=0.5,classes=[2,5,7])[0]
    for car_result in car_results.boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = car_result
        # print(f"{track_id},{class_id}")
        vehicle_bounding_boxes = []
        vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
        for bbox in vehicle_bounding_boxes: 
            roi = input_image[int(y1):int(y2), int(x1):int(x2)] # crop the vehicle
            license_plates = license_detector(roi,conf=0.5)[0]
            for license_plate in license_plates.boxes.data.tolist():
                print(track_id)
                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                # crop license plate
                plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                # de-colorize
                plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                # posterize
                _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                ocr_detections = plate_reader.readtext(plate_treshold)
                lic_text, lic_score = read_license_plate(ocr_detections)
                results[0][track_id] = {
                            'car': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score
                            },
                            'license_plate': {
                                'bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                                'bbox_score': plate_score,
                                'number': lic_text,
                                'text_score': lic_score
                            }
                } 

    return results