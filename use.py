import keras.api.models as models
from PIL import Image
from util import aiutil
import tensorflow as tf

# Direcciones de los archivos
model_path = "./model/mish/best_model.keras"
exam_path = "./images/Prueba.jpg"

# Cargamos el modelo ya entrenado
model = models.load_model(model_path)

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
    "B"
]

# Cargamos el examen y la convertimos a escala de grises
def_size = (1128, 1226)  # Tamaño por defecto
img = Image.open(exam_path)
# Reducimos el ruido y convertimos en escala de grises
img = aiutil.smooth_img(img)
# Obtenemos los contornos
filter_contours, contours = aiutil.get_countours(img, min_h=600)
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
