from PIL import Image
import io
import numpy as np
import cv2 as cv
import easyocr
from ultralytics import YOLO
from utils.license_format import license_complies_format,format_license
import yaml

with open("/Users/geuntakroh/workspace/github/license-plate-detection/inference/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# TOKEN = config["detectors"]["server_token"]
MODEL_ENDPOINT = config["detectors"]["server_url"]
VEHICLE_MODEL_NAME = config["detectors"]["vehicle_detector"]
LICENSE_MODEL_NAME = config["detectors"]["license_detector"]

vehicle_detector = YOLO(MODEL_ENDPOINT + VEHICLE_MODEL_NAME, task='detect')
vehicle_tracker = YOLO(MODEL_ENDPOINT + VEHICLE_MODEL_NAME, task='detect')
license_detector = YOLO(MODEL_ENDPOINT + LICENSE_MODEL_NAME, task='detect')
plate_reader = easyocr.Reader(['en'],gpu=False)
vehicles_id = [2,3,5,7]

def vehicle_detect_bytes(image,conf: float = 0.25):
    # if getattr(vehicle_detector,'predictor',None) is not None:
    #     print('Yolo have Predictor!!!!')
    #     if getattr(vehicle_detector.predictor,'trackers',None) is not None:
    #         print('Tracker does exist!!')
    #     else:
    #         print("Tracker does not exists!!!!")
    # else:
    #     print("Yolo does not have Predictor!!!")
    results = vehicle_detector(image,conf=conf,classes=vehicles_id)
    prediction = results[0].plot()
    return_bytes = get_bytes_from_prediction(prediction,quality=95)
    return return_bytes

def license_detect_bytes(image,conf: float = 0.25):
    
    results = license_detector(image,conf=conf)
    prediction = results[0].plot()
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
    # im_bgr = prediction[0].plot()
    # im_rgb = im_bgr[...,::-1]
    im_rgb = prediction[...,::-1]
    return_image = Image.fromarray(im_rgb)
    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=quality)
    return_bytes.seek(0)
    return return_bytes

def read_license_plate(detections):

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        # verify that text is conform to a standard license plate
        if license_complies_format(text):
            # bring text into the default license plate format
            return format_license(text), score

    return None, None

# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right

#     cv.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
#     cv.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

#     cv.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
#     cv.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

#     cv.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
#     cv.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

#     cv.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
#     cv.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

#     return img

def crop_vehicle_license_then_read(input_image,vehicle_conf: float = 0.25,license_conf: float = 0.25,frame_number: int = 0):
    # frame_results = {}
    frame_results = [] # change return value type as list

    vehicle_results = vehicle_tracker.track(input_image, persist=True,conf=vehicle_conf,classes=vehicles_id)[0]
    for vehicle_result in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = vehicle_result
        # print(f"{track_id},{class_id}")
        vehicle_bounding_boxes = []
        vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
        for bbox in vehicle_bounding_boxes: 
            roi = input_image[int(y1):int(y2), int(x1):int(x2)] # crop the vehicle
            license_plates = license_detector(roi,conf=license_conf)[0]
            for license_plate in license_plates.boxes.data.tolist():
                print(track_id)
                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                # crop license plate
                plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                # de-colorize
                plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                # posterize
                _, plate_treshold = cv.threshold(plate_gray, 64, 255, cv.THRESH_BINARY_INV)
                # ocr_detections = plate_reader.readtext(plate_treshold)
                # lic_text, lic_score = read_license_plate(ocr_detections)

                ### try rgb and gray and get the best
                rgb_detections = plate_reader.readtext(plate)
                gray_detections = plate_reader.readtext(plate_treshold)
                rgb_text, rgb_score = read_license_plate(rgb_detections)
                gray_text, gray_score = read_license_plate(gray_detections)
                
                candidates = [(rgb_text,rgb_score),(gray_text,gray_score)]
                valid_candidates = [(text,score) for text,score in candidates if score is not None]
                lic_text, lic_score = max(valid_candidates,default=(None,None))
               
                # results[frame_nmr][track_id] = {
                # frame_results[track_id] = {
                #             'vehicle': {
                #                 'bbox': [x1, y1, x2, y2],
                #                 'bbox_score': score
                #             },
                #             'license_plate': {
                #                 'bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                #                 'bbox_score': plate_score,
                #                 'number': lic_text,
                #                 'text_score': lic_score
                #             }
                # } 
                # frame_results[track_id] = {        
                #     'vehicle_bbox': [x1, y1, x2, y2],
                #     'vehicle_bbox_score': score,
                #     'lp_bbox': [plate_x1 , plate_y1, plate_x2, plate_y2],
                #     'lp_bbox_score': plate_score,
                #     'lp_number': lic_text,
                #     'lp_text_score': lic_score
                # } 
                frame_results.append({
                    "frame_number": frame_number,
                    "track_id": track_id,
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "vehicle_bbox_score": score,
                    "lp_bbox": [plate_x1 , plate_y1, plate_x2, plate_y2],
                    "lp_bbox_score": plate_score,
                    "lp_number": lic_text,
                    "lp_text_score": lic_score
                })
    print(vehicle_tracker.predictor.trackers[0])
    print(len(vehicle_tracker.predictor.trackers))
    return frame_results


def reset_tracker():
    if len(vehicle_tracker.predictor.trackers) > 0:
        vehicle_tracker.predictor.trackers[0].reset()
        print(vehicle_tracker.predictor.trackers[0])
        return True
    else:
        print('tracker does not exists')
