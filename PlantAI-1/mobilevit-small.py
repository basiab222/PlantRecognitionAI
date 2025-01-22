from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import time


image_path = "./Photo_to_predict/aloes.jpeg"
image = Image.open(image_path).convert("RGB")

processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

inputs = processor(images=image, return_tensors="pt")

start_time = time.time()

outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

end_time = time.time()

print("Predicted class:", model.config.id2label.get(predicted_class_idx, "Unknown class"))
print(f"Czas predykcji: {end_time - start_time:.4f} sekundy")
