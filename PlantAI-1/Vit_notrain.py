
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import time

image_path = "./Photo_to_predict/aloes.jpeg"
image = Image.open(image_path).convert("RGB")


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")

start_time = time.time()

outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()


end_time = time.time()

print("Predicted class:", model.config.id2label[predicted_class_idx])
print(f"Czas predykcji: {end_time - start_time:.4f} sekundy")