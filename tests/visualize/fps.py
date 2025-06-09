import cv2 as cv


ann_video_path = "../annotated_video.mp4"
video_path = "../../inputs/sample.mp4"
cut_video =  "../../inputs/sample_1st_half.mp4"

ann_video = cv.VideoCapture(ann_video_path)
video = cv.VideoCapture(video_path)
cut_video = cv.VideoCapture(cut_video)

print(f"ann video frame : {ann_video.get(cv.CAP_PROP_FPS)}")
print(f"ann video frame type: {ann_video.get(cv.CAP_PROP_FRAME_TYPE)}")
print(f"ann video frame count: {ann_video.get(cv.CAP_PROP_FRAME_COUNT)}")
print(f"video frame : {video.get(cv.CAP_PROP_FPS)}")
print(f"video frame type: {video.get(cv.CAP_PROP_FRAME_TYPE)}")
print(f"video frame count: {video.get(cv.CAP_PROP_FRAME_COUNT)}")
print(f"cut video frame : {cut_video.get(cv.CAP_PROP_FPS)}")
print(f"cut video frame type : {cut_video.get(cv.CAP_PROP_FRAME_TYPE)}")
print(f"cut video frame count: {cut_video.get(cv.CAP_PROP_FRAME_COUNT)}")