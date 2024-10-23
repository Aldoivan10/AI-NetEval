from PIL import Image
from util import aiutil

img_path = "./images/Prueba.jpg" # Ruta de la imagen
def_size = (1128, 1226)   # Tamaño por defecto

# Leemos la imágen y estandarizamos su tamaño
img = Image.open(img_path).resize(def_size)
# Obtenemos los contornos
countours = aiutil.get_countours(img, min_h=800)
# Dibujamos los contornos (Solo para pruebas)
aiutil.draw_cnts(img, countours)
# Mostramos la imagen (Solo para pruebas)
img.show()