""" import keras.api.models as  models """
from PIL import Image
from util import aiutil

# Direcciones de los archivos
model_path = "./model/mish/best_model.keras"
exam_path = "./images/Prueba.jpg"

""" # Cargamos el modelo ya entrenado
model = models.load_model(model_path) """

# Cargamos el examen y la convertimos a escala de grises
def_size = (1128, 1226)          # Tamaño por defecto
img = Image.open(exam_path).convert("L")
# Reducimos el ruido y convertimos en escala de grises
img = aiutil.smooth_img(img)
# Obtenemos los contornos
filter_contours, contours = aiutil.get_countours(img, min_h=600)
# Obtenemos las imágenes
responses = aiutil.get_responses(img, filter_contours)
# Convertimos las imágenes de las respuestas en un tensorflow dataset
dataset = aiutil.to_dataset(responses)
