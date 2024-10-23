from PIL import Image
from typing import List
from cv2.typing import MatLike, Rect, Point, Vec3i
import imutils.contours as cnts
from PIL.ImageDraw import Draw as drawer
import numpy as np

import cv2
import imutils

# Función para generar una sequencia
def infinite_sequence():
    num = 1
    while True:
        yield num
        num += 1

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

# Función para obtener los contornos ordenados de una imagen
def get_countours(img: Image.Image,  min_h: int = 100) -> List[MatLike]:
    # Convertimos a escala de grises
    img_process = img.convert('L')
    # Convertimos en arreglo numpy
    img_process = np.array(img_process)
    # Aplicamos un filtro para reducir el ruido
    img_process = cv2.bilateralFilter(img_process, d=9, sigmaColor=75, sigmaSpace=75)
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
    contours = [(poly, rect) for poly, rect in contours if rect[-1] >= min_h and len(poly) == 4]
    # Retornamos los contornos
    return contours

# Función para dibujar polígonos (Solo para pruebas)
def draw_poly(img: Image.Image, points: tuple[Point, ...], color: Vec3i = (255, 0, 0), width: int = 1):
    draw = drawer(img)
    draw.polygon(points, None, color, width)

# Función para dibujar texto (Solo para pruebas)
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

# Función para dibujar contornos (Solo para pruebas)
def draw_cnts(img: Image.Image, cnts: List[MatLike]):
    # Iniciamos la secuencia (Solo para indicar el orden)
    seq = infinite_sequence()
    # Por cada contorno
    for poly, rect in cnts:
        # Dibujamos el polígono
        draw_poly(img, [tuple(row) for row in poly], width=5)
        # Dibujamos las dimensiones
        draw_text(img,  f"{next(seq)}: {rect[2]} x {rect[3]}", rect, fill=(0,0,255), width=2, font_size=30)