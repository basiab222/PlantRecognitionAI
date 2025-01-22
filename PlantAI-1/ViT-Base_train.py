from datasets import load_dataset

from transformers import AutoImageProcessor, DefaultDataCollator
from transformers import AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
from evaluate import load

#ZMIENIĆ W RAZIE CO FOLDER (split_data)
dataset_dir = "/home/xanny/PycharmProjects/PlantAI/Dataset/resized_data"

dataset = load_dataset("imagefolder", data_dir=dataset_dir)

print("Dataset keys:", dataset.keys())
print("Sample train example:", dataset["train"][0])
print("Train column names:", dataset["train"].column_names)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_function(examples):
    print("Keys in preprocess_function:", examples.keys())
    if "image" not in examples:
        raise ValueError(
            "Brak klucza 'image' w przykładach danych. Sprawdź strukturę zbioru danych."
        )
    images = [image.convert("RGB") for image in examples["image"]]
    processed = processor(images, return_tensors="pt")
    processed["labels"] = examples["label"]
    return processed

batch = 16

train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=batch)
test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=batch)

num_labels = len(dataset["train"].features["label"].names)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./vit-plant-classifier",
    save_total_limit=3,
    load_best_model_at_end=True,
    dataloader_num_workers=4,
    num_train_epochs=15,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    metric_for_best_model="accuracy",
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DefaultDataCollator(return_tensors="pt")

def custom_optimizer(model):
    return AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    optimizers=(custom_optimizer(model), None),
    compute_metrics=compute_metrics
)

checkpoint_path = "./vit-plant-classifier/checkpoint-624"
try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
    print(f"Training resumed successfully from {checkpoint_path}")
except Exception as e:
    print(f"Failed to resume training from {checkpoint_path}: {e}")
    print("Starting training from scratch...")
    trainer.train()


try:
    metrics = trainer.evaluate(test_dataset)
    print("Model Evaluation Metrics:")
    print(metrics)
    if "eval_accuracy" in metrics:
        print(f"Model Accuracy: {metrics['eval_accuracy']:.2f}")
except Exception as e:
    print(f"Evaluation failed: {e}")