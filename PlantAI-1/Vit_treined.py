from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import time
import torch.nn.functional as F

checkpoint_path = "./vit-plant-classifier/checkpoint-1248"

try:
    processor = AutoImageProcessor.from_pretrained(checkpoint_path)
except OSError:
    print("Brak `preprocessor_config.json`. Zapisuję domyślny procesor.")
    default_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    default_processor.save_pretrained(checkpoint_path)
    processor = default_processor

model = AutoModelForImageClassification.from_pretrained(checkpoint_path)

image_path = "Photo_to_predict/photo_2025-01-18_17-50-32.jpg"
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class = logits.argmax(-1).item()
    confidence = probabilities[0, predicted_class].item()

end_time = time.time()

label_names = model.config.id2label
predicted_label = label_names.get(predicted_class, "Nieznana klasa")

print(f"Predykcja: {predicted_label}")
print(f"Pewność: {confidence * 100:.2f}%")
print(f"Czas predykcji: {end_time - start_time:.4f} sekundy")
