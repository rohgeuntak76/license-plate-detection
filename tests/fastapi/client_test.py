import requests
import io
import streamlit as st
import cv2 as cv

api_host = "localhost:8000"

# url = "http://" + api_host + "/api/image/plates/detect/annotatedImage"
url = "http://" + api_host + "/api/image/plate_number/crop/detect/info"

with open('../../inputs/car_image.jpg','rb') as img:
    print(type(img))
    # buffer = io.BytesIO(img.read())

    files = {
            'image': ('car_image', img, 'image/jpeg'),
        }
    data = {
        'vehicle_conf': f'{0.25}',
        'license_conf': f'{0.25}'
    }
    # params = [2,3,5,7]

    # response = requests.post(url,files=files,data=data,params=params,stream=True)
    response = requests.post(url,files=files,data=data,stream=True)
# print(type(response.content))
print(response.text)
print(type(response.text))
print(response.json())
print(type(response.json()))
print(type(response.json()[0]))
# a = io.BytesIO(response.content)
# print(type(a.getvalue()))
# input_image = cv.imdecode(response.content, cv.IMREAD_COLOR) 

# st.image(response.content)

#  annotated_result = io.BytesIO(response.content)

