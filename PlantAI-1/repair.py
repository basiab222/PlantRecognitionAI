from transformers import AutoImageProcessor

checkpoint_path = "./vit-plant-classifier/checkpoint-1248"

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor.save_pretrained(checkpoint_path)
