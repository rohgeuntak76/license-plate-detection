from PIL import Image
import io
import numpy as np

def get_bytes_from_prediction(prediction: np.ndarray,quality: int) -> bytes:
    """
    Convert YOLO's prediction to Bytes
    
    Args:
    prediction (np.ndarray): A YOLO's prediction
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 95
    """
    im_bgr = prediction[0].plot()
    im_rgb = im_bgr[...,::-1]
    return_image = Image.fromarray(im_rgb)
    return_bytes = io.BytesIO()
    return_image.save(return_bytes, format='JPEG', quality=quality)
    return_bytes.seek(0)
    return return_bytes