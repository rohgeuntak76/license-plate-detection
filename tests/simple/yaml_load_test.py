import yaml

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

TOKEN = config["detectors"]["server_token"]
MODEL_ENDPOINT = config["detectors"]["server_url"]
VEHICLE_MODEL_NAME = config["detectors"]["vehicle_detector"]
LICENSE_MODEL_NAME = config["detectors"]["license_detector"]

if TOKEN is not None and len(TOKEN) > 0:
    print("Token is in")
    TOKEN = TOKEN.strip()
print(len(TOKEN))
# print(TOKEN)
# print(type(TOKEN))
# print(str(TOKEN))
# print(type(str(TOKEN)))
# print(MODEL_ENDPOINT)
# print(VEHICLE_MODEL_NAME)
# print(LICENSE_MODEL_NAME)
