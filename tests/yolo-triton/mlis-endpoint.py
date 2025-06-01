from ultralytics import YOLO
import yaml
from urllib.parse import urlparse

with open("/Users/geuntakroh/workspace/github/license-plate-detection/inference/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# TOKEN = config["detectors"]["server_token"]
MODEL_ENDPOINT = config["detectors"]["server_url"]
VEHICLE_MODEL_NAME = config["detectors"]["vehicle_detector"]
LICENSE_MODEL_NAME = config["detectors"]["license_detector"]


print(MODEL_ENDPOINT)


def convert_kserve_url_to_triton_url(mlis_url):
    parts = urlparse(mlis_url)
    split = parts.netloc.split('.')[0]
    svc_name, namespace = split.split('-predictor-')
    url_for_yolo = f"http://{svc_name}-predictor.{namespace}.svc.cluster.local"
    return url_for_yolo

url_for_yolo =  convert_kserve_url_to_triton_url(MODEL_ENDPOINT)
print(url_for_yolo)