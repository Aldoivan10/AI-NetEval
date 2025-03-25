from cv2.typing import MatLike, Rect, Point, Vec3i
from PIL.ImageDraw import Draw as drawer
from typing import List
from PIL import Image

import math
import imutils
import numpy as np
from keras import Sequential
from tensorflow import Tensor
from contextlib import redirect_stdout
from tensorflow._api.v2.data import Dataset
import matplotlib.pyplot as plt
from numpy import argmax
import keras.api.preprocessing as preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

# Función para generar una sequencia
def infinite_sequence(start: int = 1):
    num = start
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
def to_img(image: Tensor):
    img = imutils.opencv2matplotlib(image.numpy().astype("uint8"))
    return Image.fromarray(img)

# Función para mostrar el mínimo y máximo valor de una imagen
def print_min_max(dataset: Dataset, name = "Dataset"):
    images, _ = next(iter(dataset))
    img = images[0]
    print(f"{name}: Min: {np.min(img)}, Max: {np.max(img)}")

# Función para guardar el modelo
def save_summary(path: str, model: Sequential):
    with open(f"{path}/summary.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary()

# Funcion para evaluar el modelo (F1 score y recall)
def evaluate_model(model: Sequential, path: str):
    test_ds = preprocessing.image_dataset_from_directory("images/Test", color_mode="grayscale", shuffle=False)
    # Obtenemos las predicciones
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.extend(labels)
        y_pred.extend(argmax(model(images), axis=1))
    
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    with open(f"{path}/summary.txt", "a") as f:
        f.write(f" Recall: {precision}\n Precision: {recall}\n F1-score-: {f1}")

# Funcion para agregar las imágenes a un plot
def show_responses(batch: Tensor, predictions: list[str], cols: int = 10):
    # tamaño de la figura
    plt.figure(figsize=(10, 10))
    # Calcular numero de filas
    rows = math.ceil(len(batch) // cols) + 1
    # Iterar sobre las imágenes y sus predicciones
    for i in range(len(batch)):
        # Ajusta la cuadrícula de subgráficas
        plt.subplot(rows, cols, i + 1)
        # Muestra la imagen 
        plt.imshow(batch[i], cmap="gray")
        # Muestra la predicción en el título 
        plt.title(predictions[i]) 
        # Ocultar los ejes
        plt.axis('off')
    # Ajusta los espacios entre subgráficas
    plt.tight_layout()
    # Muestra la figura con las imágenes y sus predicciones
    plt.show()