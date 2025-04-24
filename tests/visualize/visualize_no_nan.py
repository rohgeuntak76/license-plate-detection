import ast
from glob import glob
import cv2 as cv
import numpy as np
import pandas as pd

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

results = pd.read_csv('./missing_data_added.csv')

# load video
videos = glob('inputs/sample.mp4')
print(videos)
video = cv.VideoCapture(videos[0])

# cap = cv2.VideoCapture(video_path)

fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = video.get(cv.CAP_PROP_FPS)

print(fps)
# exit()
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter('./out_after_missing_data_added_no_nan.mp4', fourcc, fps, (width, height))

frame_number = -1
video.set(cv.CAP_PROP_POS_FRAMES, 0)

while video.isOpened():
    success, frame = video.read()
    frame_number += 1
    print(frame_number)
    if success:
        df_ = results[results['frame_number'] == frame_number]
        for index in range(len(df_)):
            # draw car
            if str(df_.iloc[index]['license_plate_number']) != 'nan':
                print(df_.iloc[index]['license_plate_number'])
                vhcl_x1, vhcl_y1, vhcl_x2, vhcl_y2 = ast.literal_eval(df_.iloc[index]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                
                draw_border(
                    frame, (int(vhcl_x1), int(vhcl_y1)),
                    (int(vhcl_x2), int(vhcl_y2)), (0, 255, 0),
                    12, line_length_x=200, line_length_y=200)
                
                # draw license plate
                plate_x1, plate_y1, plate_x2, plate_y2 = ast.literal_eval(df_.iloc[index]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

                # region of interest
                roi = frame[int(vhcl_y1):int(vhcl_y2), int(vhcl_x1):int(vhcl_x2)]
                cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 0, 255), 6)

                # write detected number
                (text_width, text_height), _ = cv.getTextSize(
                    str(df_.iloc[index]['license_plate_number']),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2,
                    6)

                cv.putText(
                    frame,
                    str(df_.iloc[index]['license_plate_number']),
                    (int((vhcl_x2 + vhcl_x1 - text_width)/2), int(vhcl_y1 - text_height)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    6
                )

        out.write(frame)
        frame = cv.resize(frame, (1280, 720))
    else:
        break

out.release()
video.release()