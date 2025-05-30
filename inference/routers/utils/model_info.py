from fastapi import APIRouter
import yaml

router = APIRouter(
    prefix="/api/utils",
)

@router.get("/models/info", tags=["Utils"])
def return_model_info():
    """
    Return Model info from config.yaml
        
    Returns:
        Dict: { "VEHICLE_MODEL_NAME" : VEHICLE_MODEL_NAME, "LICENSE_MODEL_NAME": LICENSE_MODEL_NAME}
    """
    with open("/Users/geuntakroh/workspace/github/license-plate-detection/inference/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # TOKEN = config["detectors"]["server_token"]
    MODEL_ENDPOINT = config["detectors"]["server_url"]
    VEHICLE_MODEL_NAME = config["detectors"]["vehicle_detector"]
    LICENSE_MODEL_NAME = config["detectors"]["license_detector"]
    return { "VEHICLE_MODEL_NAME" : MODEL_ENDPOINT + VEHICLE_MODEL_NAME, "LICENSE_MODEL_NAME": MODEL_ENDPOINT + LICENSE_MODEL_NAME}