import boto3
import cv2
import re
from configurations import AWS_REGION

textract = boto3.client('textract', region_name=AWS_REGION)

def ocr_plate_image(cropped_image):
    try:
        success, img_encoded = cv2.imencode('.jpg', cropped_image)
        if not success:
            return "UNKNOWN"
        img_bytes = img_encoded.tobytes()

        response = textract.detect_document_text(Document={'Bytes': img_bytes})

        raw_text = " ".join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
        cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
        matches = re.findall(pattern, cleaned_text)
        return matches[0] if matches else "Not Detected"
    except Exception as e:
        print(f"[OCR Error] {e}")
        return "UNKNOWN"
