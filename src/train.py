from util import aitest
from util import aiutil
from config.settings import config

import json
import keras
import tensorflow as tf

# Obtenemos un dataset de entrenamiento.
train_ds, validation_ds = keras.preprocessing.image_dataset_from_directory(
    config.TRAIN_PATH,
    seed=123,
    subset="both",
    validation_split=0.2,
    color_mode="grayscale",
)
# Indicamos su tipo  (No es obligatorio)
train_ds: tf.data.Dataset
validation_ds: tf.data.Dataset

# Mostramos una imagen para validar el dataset (Solo para pruebas)
img = aiutil.get_first_img(train_ds)
aitest.to_img(img).show()
# Obtenemos los nombres de las clasificaciones
class_names = train_ds.class_names
# Optimizamos la lectura y preparación de los datos
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
# Obtenemos una imagen de muestra del dataset de entrenamiento
img = aiutil.get_first_img(train_ds)
# Creamos el modelo
model = keras.Sequential(
    [
        # Funciones de data augmented
        keras.layers.Input(shape=img.shape),
        # Capa para randomizar una traslación
        keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=255),
        # Capara para randomizar una rotación
        keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=255),
        # Capa para randomizar un zoom
        keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=255),
        keras.layers.RandomBrightness(0.1),  # Capa para randomizar el brillo
        # Agregamos una capa para randomizar un contraste
        keras.layers.RandomContrast(0.1),
        # Capa de normalización, pasar valores de 0-255 a 0-1
        keras.layers.Rescaling(1.0 / 255),
    ]
    if config.DATA_AUGMENTED
    else [keras.layers.Input(shape=img.shape)]
    + [
        keras.layers.Conv2D(32, 3, activation=config.ACTIVATION_FUNCTION),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, 3, activation=config.ACTIVATION_FUNCTION),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(128, 3, activation=config.ACTIVATION_FUNCTION),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(256, 3, activation=config.ACTIVATION_FUNCTION),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=config.ACTIVATION_FUNCTION),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)
# Compilamos el modelo
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Guardamos la estructura del modelo (Solo para revisiones)
aitest.save_summary(config.MODEL_PATH, model)
# Callback para guardar el mejor modelo
best = keras.callbacks.ModelCheckpoint(
    config.MODEL_PATH / "best_model.keras", save_best_only=True
)
# Callback para detener el entrenamiento cuando el error de validación se reduzca
stopping = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
# Callback para guardar el log (Solo para revisiones, se puede omitir)
logger = keras.callbacks.CSVLogger(config.MODEL_PATH / "train.log")
# Número de epocas
epochs = 50
# Entrenamos el modelo
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    validation_batch_size=32,
    callbacks=[stopping, logger, best],
)
# Guardamos el historial (Solo para revisiones)
with open(config.MODEL_PATH / "history.json", "w") as file:
    json.dump(history.history, file)
# Guardamos la evaluacion (Solo para revisiones)
# aitest.evaluate_model(model, saved_path)
