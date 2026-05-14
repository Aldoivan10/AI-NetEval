from pathlib import Path
from PIL._typing import _Ink
from numpy.typing import NDArray
from cv2.typing import MatLike, Rect, Point
from typing import List, Tuple, AnyStr, Iterable

import math
import imutils
import contextlib
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
from keras import Sequential


# Función para generar una sequencia
def infinite_sequence(start: int = 1):
    num = start
    while True:
        yield num
        num += 1


# Función para dibujar polígonos
def draw_poly(
    img: Image.Image,
    points: Tuple[Point] | List[Point],
    color: _Ink = (255, 0, 0),
    width: int = 1,
):
    draw = ImageDraw.Draw(img)
    draw.polygon(points, None, color, width)


# Función para dibujar texto
def draw_text(
    img: Image.Image,
    text: AnyStr,
    container: Rect | MatLike,
    font_size: int = 12,
    fill: _Ink = (255, 0, 0),
    width: int = 1,
) -> None:
    draw = ImageDraw.Draw(img)
    # Obtenemos [x, y, ancho, alto] del contenedor
    [cx, cy, cw, ch] = container
    # Obtenemos el ancho y alto del texto
    [_, _, tw, th] = draw.textbbox(
        (0, 0), text, stroke_width=width, font_size=font_size
    )
    # A las dimenciones obtenemos la mitad
    tw, th = tw // 2, th // 2
    # Obtenemos el centro del padre
    [px, py] = ((cx + cw / 2), (cy + ch / 2))
    # Calculamos el nuevo origen centrado
    orig = (px - tw, py - th)
    # Dibujamos un outline
    draw.text(orig, text, (255, 255, 255), stroke_width=width * 3, font_size=font_size)
    # Dibujamos el texto
    draw.text(orig, text, fill, stroke_width=width, font_size=font_size)


# Función para dibujar contornos
def draw_cnts(img: Image.Image, cnts: List[Tuple[MatLike, MatLike]]):
    # Iniciamos la secuencia (Solo para indicar el orden)
    seq = infinite_sequence()
    # Por cada contorno
    for poly, rect in cnts:
        # Dibujamos el polígono
        draw_poly(img, [tuple(row) for row in poly], width=5)
        # Dibujamos las dimensiones
        draw_text(
            img,
            f"{next(seq)}: {rect[2]} x {rect[3]}",
            rect,
            fill=(0, 0, 255),
            width=2,
            font_size=30,
        )


# Función para mostrar imágenes de un dataset
def to_img(image: tf.Tensor) -> Image.Image:
    img = imutils.opencv2matplotlib(image.numpy().astype("uint8"))
    return Image.fromarray(img)


# Función para mostrar el mínimo y máximo valor de una imagen
def print_min_max(dataset: Iterable, name="Dataset"):
    images, _ = next(iter(dataset))
    img = images[0]
    print(f"{name}: Min: {np.min(img)}, Max: {np.max(img)}")


# Función para guardar el modelo
def save_summary(path: Path, model: Sequential):
    with open(path / "summary.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            model.summary()


# Funcion para evaluar el modelo
def evaluate_model(
    model: Sequential,
    predictions: NDArray,
    class_names: NDArray,
    dataset: tf.data.Dataset,
    path: Path,
):
    # Obtenemos las predicciones
    y_pred = np.argmax(predictions, axis=1)
    # Obtenemos los valores verdaderos
    y_test = np.concatenate([y.numpy() for _, y in dataset], axis=0)

    # Se crea la matríz de confusión
    confussion_matrix = metrics.confusion_matrix(y_test, y_pred)
    # Se crea el reporte de evaluación
    report = metrics.classification_report(y_test, y_pred, zero_division=0)

    # Se grafica la matríz de confusión
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names
    )
    # Se guarda la matríz de confución
    plt.savefig(path / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    # Se limpia el layout
    plt.close()

    # Se garda la estructura del modelo
    save_summary(path, model)
    # Se agregan las evaluaciones
    with open(path / "summary.txt", "a") as f:
        f.write(f"\nConfusion matrix: \n\n {confussion_matrix}")
        f.write(f"\n\n {report}")


# Funcion para agregar las imágenes a un plot
def show_responses(batch: List[MatLike], predictions: NDArray, cols: int = 10):
    # Tamaño de la figura
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
        plt.axis("off")
    # Ajusta los espacios entre subgráficas
    plt.tight_layout()
    # Muestra la figura con las imágenes y sus predicciones
    plt.show()


# Función para guardar todas las predicciones realizadas
def save_predictions(dataset: tf.data.Dataset, predictions: NDArray, path: Path):
    # Obtenemos las imágenes en un solo batch
    unbatched = dataset.unbatch()
    # Total de imágenes
    total = sum(1 for _ in unbatched)
    # Se calcula el total de columnas
    cols = math.ceil(math.sqrt(total))
    # Se calcula el total de filas
    rows = math.ceil(total / cols)
    # Indice de la imagen actual
    index = 1

    # Se inicializa la figura
    plt.figure(figsize=(3 * cols, 3 * rows))

    # Por cada imagen-predicción
    for (img, _), pred in zip(dataset.unbatch(), predictions):
        # Ajusta la cuadrícula de subgráficas
        plt.subplot(rows, cols, index)
        # Se muestra la imágen
        plt.imshow(img.numpy().squeeze(), cmap="gray")
        # Se muestra su predicción
        plt.title(pred, fontsize=30)
        # Ocultar los ejes
        plt.axis("off")
        # Se aumenta el índice
        index = index + 1

    # Ajusta los espacios entre subgráficas
    plt.tight_layout()
    # Se guarda las predicciones
    plt.savefig(path / "predictions.png", dpi=300, bbox_inches="tight")
    # Se limpia el layout
    plt.close()
