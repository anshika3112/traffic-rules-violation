import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
from configurations import ROBOFLOW_API_KEY

ROBO_CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

def detect_helmet(cropped_img):
    try:
        rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        result = ROBO_CLIENT.infer(pil_img, model_id="helmet-detection-ar0n2/1")
        return any(pred['class'].lower() == "helmet" and pred['confidence'] > 0.5 for pred in result['predictions'])
    except Exception as e:
        print(f"[Helmet Detection Error] {e}")
        return False
