import keras.api.preprocessing as preprocessing
import keras.api.models as  models
from util import aitest
import numpy as np

################################################################################
### Predecir un DATASET                                                      ###
################################################################################

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
for batch, _ in test_ds:
    # Procesar el batch completo
    results = model(batch)
    # Obtener las predicciones para todo el batch
    predictions = class_names[np.argmax(results.numpy(), axis=1)]  
    # Mostramos los resultados por batch
    aitest.show_responses(batch, predictions)

################################################################################
### Predecir una IMAGEN                                                      ###
################################################################################

# Cargamos una imagen, estas imágenes deben tener el mismo tamaño
# que las de entrenamiento y estar en escala de grises
img = preprocessing.image.load_img(f"{test_path}/A/A_1.jpg", color_mode="grayscale")
# Mostramos la imagen (Solo para pruebas)
img.show()
# Conertimos la imagen en un arreglo numpy
img = np.array(img)
# Incrementammos una dimension (esta dimensión indica el número de batch)
img = np.expand_dims(img, axis=0)
# Hacemos la predicción
prediction = model(img)
# Obtenemos el número de clase que es más probable
class_index = np.argmax(prediction)
# Obtenemos el nombre de la clase
class_name = class_names[class_index]
# Mostramos el resultado (Solo para pruebas)
print(f"La clase más probable es: {class_name}")