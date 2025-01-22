import os
from transformers import AutoModelForImageClassification

val_path = "./Dataset/split_data/val"

label_names = sorted(os.listdir(val_path))

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

print("id2label:", id2label)
print("label2id:", label2id)

checkpoint_path = "./vit-plant-classifier/checkpoint-1248"

model = AutoModelForImageClassification.from_pretrained(checkpoint_path)

model.config.id2label = id2label
model.config.label2id = label2id

model.save_pretrained(checkpoint_path)

print("Mapowanie zostało dodane i zapisane.")
