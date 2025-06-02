from fastapi import APIRouter
from pydantic import BaseModel,Field
import yaml

class ModelInfo(BaseModel):
    VEHICLE_MODEL_NAME : str = Field(..., description="Vehicle detection Model")
    LICENSE_MODEL_NAME : str = Field(..., description="License Plate detection Model")
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "VEHICLE_MODEL_NAME": "./models/yolo11s.pt",
                    "LICENSE_MODEL_NAME": "./models/yolo11s_20epochs_best.pt"
                },
                {
                    "VEHICLE_MODEL_NAME": "https://[kserve isvc url]/vehicle_detector",
                    "LICENSE_MODEL_NAME": "https://[kserve isvc url]/license_detector"
                },
            ]
        }

router = APIRouter(
    prefix="/api/utils",
)

@router.get(
    "/models/info",
    tags=["Utils"],
    response_model=ModelInfo,
    summary="Return Model info",
    description="Return Model info from config.yaml"
)
def return_model_info():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    MODEL_ENDPOINT = config["detectors"]["server_url"]
    VEHICLE_MODEL_NAME = config["detectors"]["vehicle_detector"]
    LICENSE_MODEL_NAME = config["detectors"]["license_detector"]
    
    return ModelInfo(VEHICLE_MODEL_NAME=MODEL_ENDPOINT + '/' + VEHICLE_MODEL_NAME, LICENSE_MODEL_NAME=MODEL_ENDPOINT + '/' + LICENSE_MODEL_NAME)