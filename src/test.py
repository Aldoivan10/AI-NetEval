from typing import Any
from numpy.typing import NDArray
from config.settings import config

import numpy as np
from util import aitest
from keras import models, preprocessing

# Cargamos el modelo ya entrenado
model: Any = models.load_model(config.MODEL)
# Cargamos el dataset de prueba (No olvidar que debe estar en escala de grises)
test_ds: Any = preprocessing.image_dataset_from_directory(
    config.TEST_PATH, color_mode="grayscale", shuffle=False
)

if test_ds and model:
    # Obtenemos los nombres de las clases
    class_names = np.array(test_ds.class_names + ["X"])

    ################################################################################
    ### Predecir un DATASET                                                      ###
    ################################################################################

    # Se realizan las predicciones
    predictions = model.predict(test_ds)
    # Se obtienen las respuestas
    results: NDArray = class_names[np.argmax(predictions, axis=1)]

    # Se avalua el modelo, recall, f1, etc
    aitest.evaluate_model(model, predictions, class_names, test_ds, config.MODEL_PATH)
    # Se guardan las predicciones como imágen
    aitest.save_predictions(test_ds, results, config.MODEL_PATH)

    ################################################################################
    ### Predecir un DATASET por batches                                          ###
    ################################################################################

    # Obtenemos las respuestas, por cada conjunto en el dataset
    for batch, _ in test_ds:
        # Procesar el batch completo
        predictions = model(batch)
        # Obtener las predicciones para todo el batch
        results = class_names[np.argmax(results, axis=1)]
        # Mostramos los resultados por batch
        aitest.show_responses(batch, results)

    ################################################################################
    ### Predecir una IMAGEN                                                      ###
    ################################################################################

    # Cargamos una imagen, estas imágenes deben tener el mismo tamaño
    # que las de entrenamiento y estar en escala de grises
    img = preprocessing.image.load_img(
        config.TEST_PATH / "A" / "A_1.jpg", color_mode="grayscale"
    )
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
