import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Ścieżka do danych
data_dir = "./Dataset/resized_data"

# Parametry danych
img_width, img_height = 150, 150  # Rozmiar obrazów do przeskalowania
batch_size = 32

# Generowanie danych treningowych i walidacyjnych
train_data_gen = ImageDataGenerator(
    rescale=1.0/255,  # Normalizacja (wartości 0-1)
    validation_split=0.2,  # Podział na dane treningowe i walidacyjne
    horizontal_flip=True,  # Losowe odbicie w poziomie
    zoom_range=0.2  # Losowe powiększenie
)

train_generator = train_data_gen.flow_from_directory(
    directory=data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"  # Dane treningowe
)

validation_generator = train_data_gen.flow_from_directory(
    directory=data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"  # Dane walidacyjne
)

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Wyjście dla liczby klas
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint - zapis najlepszych wag modelu
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Trenowanie modelu
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Liczba epok
    callbacks=[checkpoint]
)
