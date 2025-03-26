""" import keras.api.models as  models """
from PIL import Image

# Direcciones de los archivos
model_path = "./model/mish/best_model.keras"
exam_path = "./images/Prueba.jpg"

""" # Cargamos el modelo ya entrenado
model = models.load_model(model_path) """

# Cargamos el examen y la convertimos a escala de grises
def_size = (1128, 1226)          # Tamaño por defecto
img = Image.open(exam_path).convert("L")

