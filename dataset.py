from PIL import Image
from util import aiutil, aitest

img_path = "./images/Prueba.jpg" # Ruta de la imagen
def_size = (1128, 1226)          # Tamaño por defecto

# Leemos la imágen y estandarizamos su tamaño
img = Image.open(img_path).resize(def_size)
# Imagen auxiliar para dibujar lo obtenido (Solo para pruebas)
temp = img.copy()
# Reducimos el ruido y convertimos en escala de grises
img = aiutil.smooth_img(img)
# Obtenemos los contornos
filter_contours, contours = aiutil.get_countours(img, min_h=600)
# Dibujamos los contornos (Solo para pruebas)
aitest.draw_cnts(temp, contours)
# Mostramos la imagen (Solo para pruebas)
temp.show()

save_path = "./Train"

# Obtenemos las imágenes
responses = aiutil.get_responses(img, filter_contours)

# Por cada columna
for c, rows in enumerate(responses):
    # Guardamos las imágenes (Solo para el entrenamiento)
    for i, image in enumerate(rows):
        # Dibujamos un marco a la imágen para reducir lineas no deseadas
        aiutil.draw_frame(image)
        # Creamos una imagen cuadrada de 256 x 256
        image = aiutil.square_img(image)
        # Se guarda la respuesta para posteriormente clasificarla en subcarpetas
        image.save(f"{save_path}/{c}_{i}.jpg")