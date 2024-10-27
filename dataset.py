from PIL import Image
from util import aiutil, aitest

img_path = "./images/Examen.png" # Ruta de la imagen
def_size = (1128, 1226)          # Tamaño por defecto

# Leemos la imágen y estandarizamos su tamaño
img = Image.open(img_path).resize(def_size)
# Reducimos el ruido
img = aiutil.smooth_img(img)
# Imagen auxiliar para dibujar lo obtenido (Solo para pruebas)
temp = img.copy()
# Obtenemos los contornos
countours = aiutil.get_countours(img, min_h=600)
# Dibujamos los contornos (Solo para pruebas)
aitest.draw_cnts(temp, countours)
# Mostramos la imagen (Solo para pruebas)
temp.show()

save_path = "./Train"

# Obtenemos las imágenes
responses = aiutil.get_responses(img, countours)

# Por cada columna
for c, rows in enumerate(responses):
    # Guardamos las imágenes (Solo para el entrenamiento)
    for i, image in enumerate(rows):
        # Dibujamos un marco a la imágen para reducir lineas no deseadas
        aiutil.draw_frame(image)
        # Creamos una imagen cuadrada de 256 x 256
        image = aiutil.square_img(image)
        # Binarizamos la imagen
        image = aiutil.binarize_img(image)
        # Se guarda la respuesta para posteriormente clasificarla en subcarpetas
        image.save(f"{save_path}/{c}_{i}.jpg")