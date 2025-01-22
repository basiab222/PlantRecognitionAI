import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")


test_data_path = r"C:\Users\kdemc\Documents\NAI\PlantAI-20250118T190147Z-001\PlantAI\Dataset\resized_data\test"


if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Critical Error: The directory '{test_data_path}' does not exist. "
                            f"Please ensure the directory is correctly set up.")


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(test_data_path, transform=test_transforms)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)


num_classes = len(test_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load model to device
model = model.to(device)

model_weights_path = r"C:\path\to\your\model_weights.pth"

if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("Model weights loaded successfully.")
else:
    print("Warning: Pre-trained model weights not found. Using default ResNet-50 weights.")

model.eval()

all_labels = []
all_predictions = []


with torch.no_grad():
    # Testing loop
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)


        outputs = model(inputs)


        _, predicted = torch.max(outputs, 1)


        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("\nClassification Metrics:")
print("-" * 30)

precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print(f"Macro Precision: {precision:.2f}")
print(f"Macro Recall: {recall:.2f}")
print(f"Macro F1-Score: {f1:.2f}")

print("\nDetailed Per-Class Metrics:")
class_report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)
print(class_report)

# Calculate accuracy
correct = sum([p == t for p, t in zip(all_predictions, all_labels)])
total = len(all_labels)
accuracy = (correct / total) * 100
print(f"\nTest Accuracy: {accuracy:.2f} %")


print("\nClasses in test dataset:")
for idx, class_name in enumerate(test_dataset.classes):
    print(f"{idx}: {class_name}")
