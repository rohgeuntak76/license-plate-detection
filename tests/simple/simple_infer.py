from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO("yolo11n.pt")
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
results = model(source="./car_image.jpg")

bgr = results[0].plot()
print(bgr.shape)
exit()
print(type(results))
print(results[0].plot().shape())
