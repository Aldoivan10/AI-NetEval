import keras.api as keras
import tensorflow as tf
from util import aitest, aiutil
import os
import json

# Definicion de variables
test_dir = './images/Test'
train_dir = './images/Train'

# Obtenemos un dataset de entrenamiento.
train_ds, validation_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed = 123,
    subset="both",
    validation_split=0.2,
    color_mode="grayscale")

# Obtenemos un dataset para la prueba del entrenamiento
test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=123,
    color_mode="grayscale",
    pad_to_aspect_ratio=True)

# Aplicar la función de preprocesamiento a cada imagen en el lote
def apply_padding_to_batch(images, labels):
    return tf.map_fn(lambda img: aiutil.add_white_padding(img, labels), images)

# Indicamos su tipo  (No es obligatorio)
train_ds: tf.data.Dataset
test_ds: tf.data.Dataset
validation_ds: tf.data.Dataset

# Mostramos una imagen para validar el dataset (Solo para pruebas)
img = aiutil.get_first_img(train_ds)
""" aitest.to_img(img).show() """

# Obtenemos los nombres de las clasificaciones
class_names = train_ds.class_names

# Iprimimos el mínimo y máximo de una imágen de cada dataset
# Solo para validar que tiene valores de solo 0 y 1
aitest.print_min_max(test_ds, 'Test')
aitest.print_min_max(train_ds, 'Train')
aitest.print_min_max(validation_ds, 'Validation')

# Optimizamos la lectura y preparación de los datos
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Configuramos la estrategia de distribución para trabajar con hilos (Sí es posible)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Obtenemos una imagen de muestra del dataset de entrenamiento
img = aiutil.get_first_img(train_ds)
# Funcion de activacion
activation_func = 'mish'
# Creamos el modelo
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Input(shape=img.shape),
        keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=255),
        keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=255),
        keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=255),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.Rescaling(1./255),
        
        keras.layers.Conv2D(32, (3, 3), activation=activation_func),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation=activation_func),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation=activation_func),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(256, (3, 3), activation=activation_func),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=activation_func),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])
    # Compilamos el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Numero de intento
try_num = 3
# Ruta para guardar los archivos
saved_path = f'./model/{activation_func}_{try_num}'
os.makedirs(saved_path, exist_ok=True)

# Callback para guardar el mejor modelo
best = keras.callbacks.ModelCheckpoint(f'{saved_path}/best_model.keras', save_best_only=True)
# Callback para detener el entrenamiento cuando el error de validación se reduzca
stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# Callback para guardar el log (Solo para revisiones, se puede omitir)
logger = keras.callbacks.CSVLogger(f"{saved_path}/train.log")
# Número de epocas
epochs = 50
# Entrenamos el modelo
history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, batch_size=32, validation_batch_size=32, callbacks=[stopping, logger, best])
# Guardamos el historial (Solo para revisiones)
with open(f'{saved_path}/history.json', 'w') as file:
    json.dump(history.history, file)