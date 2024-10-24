from PIL import Image
from util import aiutil, aitest

img_path = "./images/Prueba.jpg" # Ruta de la imagen
def_size = (1128, 1226)          # Tamaño por defecto

# Leemos la imágen y estandarizamos su tamaño
img = Image.open(img_path).resize(def_size)
# Imagen auxiliar para dibujar lo obtenido (Solo para pruebas)
temp = img.copy()
# Obtenemos los contornos
countours = aiutil.get_countours(img, min_h=800)
# Dibujamos los contornos (Solo para pruebas)
aitest.draw_cnts(temp, countours)
# Mostramos la imagen (Solo para pruebas)
temp.show()

save_path = "./Train"

# Obtenemos las imágenes
responses = aiutil.get_responses(img, countours)
# Guardamos las imágenes (Solo para el entrenamiento)
for c, rows in enumerate(responses):
    for i,row in enumerate(rows):
        # Se guarda la respuesta para posteriormente clasificarla en subcarpetas
        row.save(f"{save_path}/{c}_{i}.jpg")