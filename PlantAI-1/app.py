from flask import Flask, request, jsonify, render_template 
from transformers import MobileViTImageProcessor, MobileViTForImageClassification, AutoImageProcessor, AutoModelForImageClassification 
from PIL import Image  
import torch  
import torch.nn.functional as F 
import time 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():

    image_file = request.files.get('image')
    if not image_file: 
        return jsonify({'error': 'No image uploaded'}), 400

    model_choice = request.form.get('model_choice')

    image = Image.open(image_file).convert("RGB")

    if model_choice == 'MobileViT':
        processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    elif model_choice == 'Swin':
        processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
    elif model_choice == 'ViT':
        processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
    elif model_choice == 'Trained ViT': 
        checkpoint_path = "./vit-plant-classifier/checkpoint-1248"
        try:
            processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        except OSError:
            print("Brak `preprocessor_config.json`. Zapisuję domyślny procesor.")
            default_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            default_processor.save_pretrained(checkpoint_path)
            processor = default_processor
        model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    elif model_choice == 'ResNet':
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    inputs = processor(images=image, return_tensors="pt")

    start_time = time.time()

    with torch.no_grad(): 
        outputs = model(**inputs) 
        logits = outputs.logits 
        predicted_class_idx = logits.argmax(-1).item() 

        probabilities = F.softmax(logits, dim=-1)  
        confidence = probabilities[0, predicted_class_idx].item() 

    end_time = time.time()

    label_names = model.config.id2label 
    predicted_label = label_names.get(predicted_class_idx, "Unknown class") 

    return jsonify({
        'prediction': predicted_label, 
        'confidence': f'{confidence * 100:.2f}%', 
        'time_taken': f'{end_time - start_time:.4f} seconds' 
    })

if __name__ == '__main__':
    app.run(debug=True)
