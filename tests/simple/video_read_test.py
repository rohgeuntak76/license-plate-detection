import cv2 as cv
from glob import glob


# videos = glob('datasets/Videos/sample.mp4')
videos = glob('../../inputs/sample2.mp4')
print(videos)

video = cv.VideoCapture(videos[0])

print(type(video))

video.release()