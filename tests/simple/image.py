from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = '../../inputs/car_image.jpg'
# image = Image.open(path) # return Image object of PIL
# print(type(image))
# print(image.size)

# image_array = np.array(image)
# print(type(image_array))
# print(image_array.shape)

# cv_image = cv.imread(path) # return numpy array
# print(type(cv_image))
# print(cv_image.shape)

# mat_image = plt.imread(path)
# print(type(mat_image))
# print(mat_image.shape)
import sys

image = cv.imread(path) # read file -> return numpy array
print(image)
# print(image.shape)
# print(sys.getsizeof(image))
# exit()
retval, buffer = cv.imencode('.jpg', image , [int(cv.IMWRITE_JPEG_QUALITY), 95]) # get numpy array -> return encoded JPEG bytes ( complete jpeg file content ( header,metadata,compressed data..etc)) in 1D ndarray

print(type(buffer))
print(buffer.shape)
print(buffer)
# print(buffer)

img_bytes = buffer.tobytes()
# print(type(img_bytes))

# result = cv.imdecode(img_bytes,cv.IMREAD_COLOR)
result = cv.imdecode(buffer,cv.IMREAD_COLOR)
print(type(result))
print(result.shape)

if retval:
    with open('sample.jpg','wb') as f:
        # f.write(buffer)
        f.write(img_bytes)