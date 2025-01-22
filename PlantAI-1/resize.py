import os
import shutil
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Dataset", "split_data")
output_dir = os.path.join(base_dir, "Dataset", "resized_data")

new_size = (224, 224)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

def resize_images(data_dir, output_dir, new_size):
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        split_output_dir = os.path.join(output_dir, split)

        os.makedirs(split_output_dir, exist_ok=True)

        for class_name in os.listdir(split_dir):
            class_input_dir = os.path.join(split_dir, class_name)
            class_output_dir = os.path.join(split_output_dir, class_name)

            os.makedirs(class_output_dir, exist_ok=True)

            for img_name in os.listdir(class_input_dir):
                input_img_path = os.path.join(class_input_dir, img_name)
                output_img_path = os.path.join(class_output_dir, img_name)

                try:
                    with Image.open(input_img_path) as img:
                        img_resized = img.convert("RGB").resize(new_size, Image.ANTIALIAS)
                        img_resized.save(output_img_path, "JPEG")
                except Exception as e:
                    print(f"Error processing {input_img_path}: {e}")

resize_images(data_dir, output_dir, new_size)

print("Resizing completed! Resized images saved in:", output_dir)
