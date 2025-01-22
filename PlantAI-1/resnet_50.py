from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import time

image_path = "./Photo_to_predict/aloes.jpeg"
image = Image.open(image_path).convert("RGB")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(images=image, return_tensors="pt")

start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(-1).item()
end_time = time.time()

label_name = model.config.id2label.get(predicted_label, "Nieznana klasa")
print(f"Predykcja: {label_name}")
print(f"Czas predykcji: {end_time - start_time:.4f} sekundy")
