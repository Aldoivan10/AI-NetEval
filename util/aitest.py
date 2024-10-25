from cv2.typing import MatLike, Rect, Point, Vec3i
from PIL.ImageDraw import Draw as drawer
from typing import List
from PIL import Image

import imutils
import numpy as np
import tensorflow as tf

# Función para generar una sequencia
def infinite_sequence():
    num = 1
    while True:
        yield num
        num += 1

# Función para dibujar polígonos
def draw_poly(img: Image.Image, points: tuple[Point, ...], color: Vec3i = (255, 0, 0), width: int = 1):
    draw = drawer(img)
    draw.polygon(points, None, color, width)

# Función para dibujar texto
def draw_text(img: Image.Image, text: str, container: Rect, font_size: int = 12, fill: Vec3i = (255, 0, 0),  width: int = 1) -> None:
    draw = drawer(img)
    # Obtenemos [x, y, ancho, alto] del contenedor
    [cx, cy, cw, ch] = container 
    # Obtenemos el ancho y alto del texto
    [_, _, tw, th] = draw.textbbox((0,0), text, stroke_width=width, font_size=font_size) 
    # A las dimenciones obtenemos la mitad
    tw, th = tw // 2, th // 2                
    # Obtenemos el centro del padre                                             
    [px, py] = ((cx + cw / 2), (cy + ch / 2))
    # Calculamos el nuevo origen centrado
    orig = (px - tw, py - th)
    # Dibujamos un outline
    draw.text(orig, text, (255,255,255), stroke_width=width * 3, font_size=font_size)
    # Dibujamos el texto
    draw.text(orig, text, fill, stroke_width=width, font_size=font_size)

# Función para dibujar contornos
def draw_cnts(img: Image.Image, cnts: List[MatLike]):
    # Iniciamos la secuencia (Solo para indicar el orden)
    seq = infinite_sequence()
    # Por cada contorno
    for poly, rect in cnts:
        # Dibujamos el polígono
        draw_poly(img, [tuple(row) for row in poly], width=5)
        # Dibujamos las dimensiones
        draw_text(img,  f"{next(seq)}: {rect[2]} x {rect[3]}", rect, fill=(0,0,255), width=2, font_size=30)

# Función para mostrar imágenes de un dataset 
def show_img(image: tf.image):
    img = imutils.opencv2matplotlib(image.numpy().astype("uint8"))
    Image.fromarray(img).show()

# Función para mostrar el mínimo y máximo valor de una imagen
def print_min_max(dataset: tf.data.Dataset, name = "Dataset"):
    images, _ = next(iter(dataset))
    img = images[0]
    print(f"{name}: Min: {np.min(img)}, Max: {np.max(img)}")