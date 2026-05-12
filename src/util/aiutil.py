from typing import List, Tuple
from numpy.typing import NDArray

import cv2
import imutils
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from keras import layers as klayers
from imutils import contours, perspective

# Función de reescalado de valores
reescaling_func = klayers.Rescaling(1.0 / 255)


# Función para obtener la primer imágen de un dataset
def get_first_img(dataset: tf.data.Dataset) -> tf.Tensor:
    images, _ = next(iter(dataset))
    return images[0]


# Función para aproximar el contorno a un polígono, regular o irregular
def approx_contour(contour: NDArray, regular: bool = True) -> Tuple[NDArray, NDArray]:
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
    return np.squeeze(approx, axis=1), np.array(
        cv2.boundingRect(approx), dtype=np.uint16
    )


def smooth_img(img: Image.Image, sc: int = 75, sp: int = 75):
    # Convertimos a escala de grises
    img = img.convert("L")
    # Convertimos en arreglo numpy
    arr = np.array(img)
    # Aplicamos un filtro para reducir el ruido
    arr = cv2.bilateralFilter(arr, d=9, sigmaColor=sc, sigmaSpace=sp)
    # Retornamos la imagen
    return Image.fromarray(arr)


# Función para obtener los contornos ordenados de una imagen
def get_countours(
    img: Image.Image, min_h: int = 100
) -> Tuple[List[Tuple[NDArray, NDArray]], List[Tuple[NDArray, NDArray]]]:
    # Convertimos en arreglo numpy
    img_process = np.array(img)
    # Aplicamos filtro para detección de bordes
    img_process = imutils.auto_canny(img_process)
    # Buscamos los contornos
    cnts = cv2.findContours(img_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Obtenemos los contornos
    cnts = imutils.grab_contours(cnts)
    # Ordenamos los contornos
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")
    # Aproximamos cada contorno devuelve una tupla (polígono, contenedor)
    cnts = [approx_contour(cnt) for cnt in cnts]
    # Filtramos los contornos que no cumplen los criterios de alto mínimo y no sea de 4 lados
    filter_contours = [
        (poly, rect) for poly, rect in cnts if rect[-1] >= min_h and len(poly) == 4
    ]
    # Retornamos los contornos
    return filter_contours, cnts


# Función para obtener las respuestas de una columna
def get_responses(
    img: Image.Image, countours: List[Tuple[NDArray, NDArray]], rows: int = 10
) -> List[List[Image.Image]]:
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
        img_rows = [col.crop((0, h * i, w, h * (i + 1))) for i in range(rows)]
        # Agregamos las filas a la lista de respuestas
        responses.append(img_rows)
    # Retornamos la lista de respuestas
    return responses


# Funcion para convertir una lista de PIl.Image a un dataset
def to_dataset(dataset: List[List[Image.Image]]):
    # Aplanamos el arraglo
    images = [img for col in dataset for img in col]
    # Obtenemos el tamaño de la primera imagen para usarla como referencia
    size = images[0].size
    # Redimencionamos las images para que todas tegan el mismo tamaño
    images = list(map(lambda img: img.resize(size, Image.Resampling.LANCZOS), images))
    # Agregamos padding a las imagenes para hacerlas cuadradas y convertimos a escala de grises
    images = list(map(lambda img: square_img(img).convert("L"), images))
    # Convertimos las imagenes en tensores
    images = [tf.convert_to_tensor(np.array(img), dtype=tf.float32) for img in images]
    # Retornamos el dataset
    return tf.convert_to_tensor(images)


# Función para hacer cuadrada una imagen
def square_img(
    image: Image.Image,
    size: int = 256,
    pad_color: tuple[int, int, int] = (255, 255, 255),
):
    # Obtenemos el ancho y alto de la imagen
    img_w, img_h = image.size
    # Si ya es del tamaño deseado
    if img_w == size and img_h == size:
        return image
    # Creamos una imagen del tamaño deseado con el color de relleno
    new_img = Image.new("RGB", (size, size), pad_color)
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
    # Retornamos la nueva imagen
    return new_img


# Función para dibujar un marco
def draw_frame(img: Image.Image, color: int = 255, thickness: int = 20):
    # Obtenemos el tamaño
    w, h = img.size
    # Inicializamos el dibujador
    drawer = ImageDraw.Draw(img)
    # Dibujamos el marco
    drawer.line([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], fill=color, width=thickness)
