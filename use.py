import keras.api.preprocessing as preprocessing
import keras.api.models as  models
from util import aitest
import numpy as np

# Direcciones de los archivos
model_path = "./model/mish/best_model.keras"
test_path = "./images/Test"
# Cargamos el modelo ya entrenado
model = models.load_model(model_path)
# Cargamos el dataset de prueba (No olvidar que debe estar en escala de grises)
test_ds = preprocessing.image_dataset_from_directory(test_path, color_mode="grayscale")
# Obtenemos los nombres de las clases
class_names = np.array(test_ds.class_names)
# Obtenemos las respuestas, por cada conjunto en el dataset
for batch, labels in test_ds:
    # Procesar el batch completo
    results = model(batch)
    # Obtener las predicciones para todo el batch
    predictions = class_names[np.argmax(results.numpy(), axis=1)]  
    # Mostramos los resultados
    aitest.show_responses(batch, predictions)