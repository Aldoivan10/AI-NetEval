import imutils.perspective as perspective
from cv2.typing import MatLike, Rect
import imutils.contours as cnts
from PIL.ImageDraw import Draw
from typing import List
from PIL import Image
import numpy as np

import cv2
import imutils
import tensorflow as tf
import keras.api as keras

# Función de reescalado de valores
reescaling_func = keras.layers.Rescaling(1./255)

# Función para obtener la primer imágen de un dataset
def get_first_img(dataset: tf.data.Dataset) -> tf.image:
    images, _ = next(iter(dataset))
    return images[0]

# Función para aproximar el contorno a un polígono, regular o irregular
def approx_contour(contour: MatLike, regular: bool = True) -> tuple[MatLike, Rect]:
    # Obtenemos su perimetro
    perimeter = cv2.arcLength(contour, True)
    # Definimos la distancia máxima entre el contorno y el contorno aproximado
    # Factor puede ajustarse según resultados
    factor = 0.02 if regular else 0.1
    epsilon = factor * perimeter
    # Obtenemos la aproximación
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # Retornamos los resultados
    # Squeeze elimina los ejes vacios y boundingRect convierte la aproximación en un Rect
    return np.squeeze(approx, axis=1), np.array(cv2.boundingRect(approx), dtype=np.uint16)

def smooth_img(img: Image.Image, sc: int = 75, sp: int = 75):
    # Convertimos a escala de grises
    img = img.convert('L')
    # Convertimos en arreglo numpy
    img = np.array(img)
    # Aplicamos un filtro para reducir el ruido
    img = cv2.bilateralFilter(img, d=9, sigmaColor=sc, sigmaSpace=sp)
    # Retornamos la imagen
    return Image.fromarray(img)

# Función para obtener los contornos ordenados de una imagen
def get_countours(img: Image.Image,  min_h: int = 100) -> List[MatLike]:
    # Convertimos en arreglo numpy
    img_process = np.array(img)
    # Aplicamos filtro para detección de bordes
    img_process = imutils.auto_canny(img_process)
    # Buscamos los contornos
    contours = cv2.findContours(img_process, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    # Obtenemos los contornos
    contours = imutils.grab_contours(contours)
    # Ordenamos los contornos
    contours, _ = cnts.sort_contours(contours, method="left-to-right")
    # Aproximamos cada contorno devuelve una tupla (polígono, contenedor)
    contours = [approx_contour(cnt) for cnt in contours]
    # Filtramos los contornos que no cumplen los criterios de alto mínimo y no sea de 4 lados
    filter_contours = [(poly, rect) for poly, rect in contours if rect[-1] >= min_h and len(poly) == 4]
    # Retornamos los contornos
    return filter_contours, contours

# Función para obtener las respuestas de una columna
def get_responses(img: Image.Image, countours: List[MatLike], rows: int = 10) -> List[List[Image.Image]]:
    # Lista de respuestas
    responses = []
    # Convertimos la imagen en un arreglo numpy
    arr = np.array(img)
    # Obtenemos las columnas
    cols = [perspective.four_point_transform(arr, poly) for poly, _ in countours]
    # Transformamos cada columna en una imagen
    cols = [Image.fromarray(c) for c in cols]
    # Dividimos cada columna
    for col in cols: 
        # Obtenemos las dimenciones de una fila
        w, h = col.size[0], col.size[1] // rows
        # Obtenemos las filas con las respuestas
        img_rows = [col.crop([0, h * i, w, h * (i + 1)]) for i in range(rows)]
        # Agregamos las filas a la lista de respuestas
        responses.append(img_rows)
    # Retornamos la lista de respuestas
    return responses

# Funcion para convertir una lista de PIl.Image a un dataset
def to_dataset(images: List[List[Image.Image]]):
    # Aplanamos el arraglo
    images = [img for col in images for img in col]
    # Obtenemos el tamaño de la primera imagen para usarla como referencia
    size = images[0].size
    # Redimencionamos las images para que todas tegan el mismo tamaño
    images = list(map(lambda img: img.resize(size, Image.Resampling.LANCZOS), images))
    # Convertimos las imagenes en tensores
    images = [tf.convert_to_tensor(np.array(img), dtype=tf.float32) for img in images]
    # Retornamos el dataset
    return tf.data.Dataset.from_tensor_slices(images)

# Función para hacer cuadrada una imagen
def square_img(image: Image.Image, size: int = 256, pad_color: tuple[int, int, int] = (255,255,255)):
    # Obtenemos el ancho y alto de la imagen
    img_w, img_h = image.size
    # Si ya es del tamaño deseado
    if img_w == size and  img_h == size:
        return image
    # Creamos una imagen del tamaño deseado con el color de relleno
    new_img = Image.new('RGB', (size, size), pad_color)
    # Si el ancho es menor que el alto
    if img_w < img_h:
        # Redimensionamos el alto de la imagen
        image = image.resize((img_w, size))
        # Pegamos la imagen en la nueva imagen
        new_img.paste(image, ((size - img_w) // 2, 0)) 
    # El alto es menor que el ancho
    else:
        # Redimensionamos el ancho de la imagen
        image = image.resize((size, img_h))
        # Pegamos la imagen en la nueva imagen
        new_img.paste(image, (0, (size - img_h) // 2))
    #Retornamos la nueva imagen
    return new_img

# Función para dibujar un marco
def draw_frame(img: Image.Image, color: int = 255, thickness: int = 20):
    # Obtenemos el tamaño
    w, h = img.size
    # Inicializamos el dibujador
    drawer = Draw(img)
    # Dibujamos el marco
    drawer.line([(0,0), (w, 0),(w, h), (0,h), (0,0)], fill=color, width=thickness)