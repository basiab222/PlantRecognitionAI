import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Dataset", "house_plant_species")
output_dir = os.path.join(base_dir, "Dataset", "split_data")

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")


if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory does not exist: {data_dir}")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith((".png", ".jpg", ".jpeg"))]

    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, class_name))
    for img in val_imgs:
        shutil.copy(img, os.path.join(val_dir, class_name))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, class_name))

print("Podział danych zakończony!")
