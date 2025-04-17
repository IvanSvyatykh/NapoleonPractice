from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

image = Image.open("/home/ivan/NapoleonIT/OCR/data/test/images/2266_image_106154.jpg")
langs = ["en"] # Replace with your languages or pass None (recommended to use None)
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()
predictions = recognition_predictor([image], [langs],detection_predictor)
print(predictions)