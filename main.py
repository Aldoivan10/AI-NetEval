from PIL import Image
import tensorflow as tf
from src.util import aiutil
import keras.models as kmodels
from config.settings import config

# Cargamos el modelo ya entrenado
model = kmodels.load_model(config.MODEL_PATH / "best_model.keras")
if model:
    # Clases usadas
    class_names = ["A", "B", "C", "D", "X"]
    # Respuestas correctas
    template = [
        "B",
        "B",
        "C",
        "A",
        "B",
        "C",
        "D",
        "D",
        "B",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "C",
        "B",
        "B",
        "C",
        "C",
        "A",
        "D",
        "B",
        "C",
        "C",
        "C",
        "B",
        "D",
        "B",
        "D",
        "C",
        "B",
        "B",
        "C",
        "B",
        "B",
        "C",
        "C",
        "B",
        "B",
    ]
    # Cargamos el examen y la convertimos a escala de grises
    img = Image.open(config.SRC_PATH / "Prueba.jpg").resize(config.IMG_SIZE)
    # Reducimos el ruido y convertimos en escala de grises
    img = aiutil.smooth_img(img)
    # Obtenemos los contornos
    filter_contours, contours = aiutil.get_countours(
        img, min_h=config.IMG_CONTOUR_TARGET_SIZE
    )
    # Obtenemos las imágenes
    responses = aiutil.get_responses(img, filter_contours)
    # Convertimos las imágenes de las respuestas en un tensorflow dataset
    dataset = aiutil.to_dataset(responses)
    # Procesamos el dataset
    results = model(dataset)
    # Obtenemos los indices de las clases mas probables
    predictions = tf.argmax(results, axis=-1).numpy()
    # Obtenemos las respuestas
    responses = [class_names[i] for i in predictions]
    # Agrupamos las respuestas por columna (se agrupan de 10 en 10 ya que son 4 columnas de 10 preguntas cada una)
    cols = [responses[i : i + 10] for i in range(0, len(responses), 10)]
    # Imprimimos las predicciones
    print(cols)
    # Obtenemos la calificaión
    total = [1 if r == t else 0 for r, t in zip(responses, template)]
    degree = sum(total) / len(total) * 10
    print(f"Calificación: {degree}")
