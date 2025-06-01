import requests

url = "http://localhost:8000/api/image/vehicles/detect/annotated"

file_path = './car_image.jpg'

data = open(file_path,'rb')

files = {
    'image': ("car_image.jpg", data, 'image/jpeg'),
}
data = {
    'conf': '0.25'
}
headers = {
    'accept': 'application/json',
    'Content-Type': 'multipart/form-data',
}

numbers = [1, 2, 3, 4, 5]
params = [('classes', str(n)) for n in numbers]
print(params)
# exit()

response = requests.post(url,files=files,data=data,params=params)
print(response.status_code)
response_json = response.json()
print(response_json)