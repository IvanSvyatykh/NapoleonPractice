from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ray import serve
import torch

app = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 1})
@serve.ingress(app)
class TrOCRService:
    def __init__(self):
        # Загрузка модели и процессора
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @app.post("/recognize_text")
    async def recognize_text(self, file: UploadFile = File(...)):
        # Проверка типа файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        try:
            # Чтение изображения
            image_data = await file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            
            # Предобработка и распознавание
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            recognized_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {"recognized_text": recognized_text}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

    @app.get("/healthcheck")
    async def healthcheck(self):
        return {"status": "healthy"}