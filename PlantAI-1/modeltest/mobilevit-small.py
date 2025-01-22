import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import time
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Ustawienia urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja procesora i modelu dla apple/mobilevit-small
processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small")
model.eval()

# Ścieżka do danych testowych
test_data_path = r"C:\Users\kdemc\Documents\NAI\PlantAI-20250118T190147Z-001\PlantAI\Dataset\resized_data\test"
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Nie znaleziono katalogu testowego: {test_data_path}")

# Definicja transformacji
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # MobileViT wymaga większej rozdzielczości
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Ładowanie zbioru danych
test_dataset = datasets.ImageFolder(test_data_path, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mapowanie etykiet
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
valid_classes = set(idx_to_class.keys())  # Tylko klasy ze zbioru danych

print(f"Classes in test dataset: {idx_to_class}")

true_labels = []
predicted_labels = []

# Pomiar czasu testowania
start_time = time.time()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8")) for img in inputs]
        transformed_inputs = processor(images=inputs, return_tensors="pt")
        for key in transformed_inputs:
            transformed_inputs[key] = transformed_inputs[key].to(device)
        true_labels.extend(labels.numpy())
        outputs = model(**transformed_inputs)
        logits = outputs.logits
        predicted_labels.extend(logits.argmax(dim=1).cpu().numpy())

end_time = time.time()

# Ograniczenie przewidywanych klas do istniejących w idx_to_class (przed obliczeniem metryk)
predicted_labels = [
    label if label in valid_classes else -1  # Ograniczamy do znanych klas, -1 oznacza "nieznana klasa"
    for label in predicted_labels
]
if -1 in predicted_labels:
    print(f"Ostrzeżenie: Znaleziono nieznane klasy w przewidywanych wynikach. Zostały one oznaczone jako '-1'.")

# Usunięcie wyników z nieznanych klas (-1)
filtered_true_labels = [tl for tl, pl in zip(true_labels, predicted_labels) if pl != -1]
filtered_predicted_labels = [pl for pl in predicted_labels if pl != -1]

# Mapowanie klas numerycznych do nazw
true_classes = [idx_to_class[label] for label in filtered_true_labels]
predicted_classes = [idx_to_class[label] for label in filtered_predicted_labels]

# Obliczenie metryk
precision = precision_score(filtered_true_labels, filtered_predicted_labels, average='macro', zero_division=1)
recall = recall_score(filtered_true_labels, filtered_predicted_labels, average='macro', zero_division=1)
f1 = f1_score(filtered_true_labels, filtered_predicted_labels, average='macro', zero_division=1)

# Wyświetlenie metryk
print("\nMetryki klasyfikacji:")
print(f"Precyzja makro: {precision:.4f}")
print(f"Recall makro: {recall:.4f}")
print(f"F1-Score makro: {f1:.4f}")

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
class_report = classification_report(
    filtered_true_labels,
    filtered_predicted_labels,
    labels=list(valid_classes),  # Tylko znane klasy
    target_names=list(idx_to_class.values()),
    zero_division=1,
)
print(class_report)

# Dokładność
accuracy = (
                   sum([1 for true, pred in zip(filtered_true_labels, filtered_predicted_labels) if true == pred])
                   / len(filtered_true_labels)
           ) * 100
print(f"\nDokładność: {accuracy:.2f}%")

# Statystyki czasowe
print(f"\nCzas trwania testowania: {end_time - start_time:.4f} sekundy")
