from PIL import Image
from util import aitest
from util import aiutil
from config.settings import config

# Leemos la imágen y estandarizamos su tamaño
img = Image.open(config.SRC_PATH / "Prueba.jpg").resize(config.IMG_SIZE)
# Imagen auxiliar para dibujar lo obtenido (Solo para pruebas)
temp = img.copy()
# Reducimos el ruido y convertimos en escala de grises
img = aiutil.smooth_img(img)
# Obtenemos los contornos
filter_contours, contours = aiutil.get_countours(
    img, min_h=config.IMG_CONTOUR_TARGET_SIZE
)
# Dibujamos los contornos (Solo para pruebas)
aitest.draw_cnts(temp, contours)
# Mostramos la imagen (Solo para pruebas)
temp.show()
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
        image.save(config.TRAIN_PATH / f"{c}_{i}.jpg")
