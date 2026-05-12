# 📚 Documentación de Funciones

## **Función 1** `Image.convert`

```python
def convert(mode: str | None = None, 
           matrix: tuple[float, ...] | None = None, 
           dither: Dither | None = None, 
           palette: Palette = Palette.WEB, 
           colors: int = 256) → Image
```

### Parámetros

- **`mode`**: Este es el parámetro más importante y especifica a qué **[modo de color](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)** quieres convertir la imagen.

- **`matrix`**: Es opcional y se puede utilizar para especificar una matriz de transformación personalizada. Generalmente, no se usa a menos que necesites una conversión específica y personalizada de color.

- **`dither`**: Controla la cantidad de dithering (difuminado de colores) aplicada durante la conversión. [Los métodos disponibles](https://pillow.readthedocs.io/en/stable/reference/Image.html#dither-modes) son `Dither.NONE` o `Dither.FLOYDSTEINBERG` (por defecto). Tenga en cuenta que esto no se utiliza cuando se suministra la matriz.

- **`palette`**: Es opcional, relevante solo cuando se convierte a modo P (paleta de colores) debido a que define cómo se debe crear la paleta. [Las paletas](https://pillow.readthedocs.io/en/stable/reference/Image.html#palettes) disponibles son `Palette.WEB` o `Palette.ADAPTIVE`.

- **`colors`**: Es opcional y se usa para definir el número de colores cuando se está convirtiendo a una imagen en modo paleta. Por defecto es 256.

---

## **Función 2** `bilateralFilter`

```python
def bilateralFilter(src: MatLike, 
                   d: int, 
                   sigmaColor: float, 
                   sigmaSpace: float) -> MatLike
```

### Parámetros

- **`src`**: Imágen a aplicar filtro.

- **`d`**: Diámetro de cada vecindario de píxeles utilizado durante el filtrado. Si no es positivo, se calcula a partir de `sigmaSpace`.

- **`sigmaColor`**: Filtro sigma en el espacio de color. Un valor mayor de este parámetro significa que los colores más distantes dentro del vecindario del píxel se mezclarán entre sí, resultando en áreas más grandes de color semi-igual.

- **`sigmaSpace`**: Filtro sigma en el espacio de coordenadas. Un valor mayor de este parámetro significa que los píxeles más alejados se influirán mutuamente, siempre que sus colores sean lo suficientemente cercanos. Cuando `d > 0`, especifica el tamaño del vecindario independientemente de `sigmaSpace`. De lo contrario, `d` es proporcional a `sigmaSpace`.

---

## **Función 3** `auto_canny`

```python
def auto_canny(image: MatLike, sigma: float = 0.33)
```

### Parámetros

- **`image`**: Imágen a aplicar filtro.

- **`sigma`**: Argumento opcional, puede utilizarse para variar los umbrales porcentuales que se determinan basándose en estadísticas simples. Con valor por defecto de 0.33.

---

## **Función 4** `findContours`

```python
def findContours(image: MatLike, 
                mode: int, 
                method: int, 
                contours: Sequence[MatLike] | None = None, 
                hierarchy: MatLike | None = None, 
                offset: Point = (0,0))
```

### Parámetros

- **`image`**: Imagen donde se buscarán los contornos.

- **`mode`**: El [modo de recuperación de contornos](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71) que define la jerarquía de los mismos. Los valores posibles son:
  - **`RETR_EXTERNAL`**: Este método recupera solo los contornos exteriores. Es útil cuando solo te interesa detectar los límites externos de los objetos en la imagen, de modo que si hay un contorno que encierra a otro (como círculos concéntricos), sólo se da el más exterior.
  - **`RETR_LIST`**: Este método recupera todos los contornos sin establecer relaciones jerárquicas. Es útil cuando necesitas todos los contornos, pero no te importa la estructura jerárquica entre ellos.
  - **`RETR_TREE`**: Este método recupera todos los contornos y reconstruye una jerarquía completa de contornos anidados. Es útil cuando necesitas entender la relación entre contornos internos y externos, como en la detección de objetos dentro de otros objetos.
  - **`RETR_CCOMP`**: Similar a `RETR_TREE`, pero solo crea relaciones jerárquicas entre contornos de primer y segundo nivel. Es útil cuando necesitas una jerarquía limitada a dos niveles.

- **`method`**: El [método de aproximación de contornos](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff) que define cómo se representan los contornos. Los valores posibles son:
  - **`CHAIN_APPROX_NONE`**: Guarda todos los puntos del contorno. Es la más precisa, pero puede generar muchos puntos, lo que aumenta el procesamiento.
  - **`CHAIN_APPROX_SIMPLE`**: Reduce la cantidad de puntos guardando solo los puntos esenciales que forman el contorno, eliminando los puntos redundantes. Esto simplifica mucho el contorno y es útil para la mayoría de los casos.
  - **`CHAIN_APPROX_TC89_L1`** y **`CHAIN_APPROX_TC89_KCOS`**: Son métodos de aproximación de contornos que utilizan el algoritmo de Teh-Chin, que proporciona una mayor precisión y puede ser útil en casos específicos donde se necesita una mayor exactitud.

- **`contours`**: Opcional. Una lista de contornos detectados, donde cada contorno es una lista de puntos. Si se proporciona, se llena con los contornos detectados.

- **`hierarchy`**: Opcional. Una matriz que describe la relación jerárquica entre los contornos. Es útil para entender cómo se relacionan los contornos entre sí (por ejemplo, contornos internos dentro de contornos externos).

- **`offset`**: Opcional. Un desplazamiento para ajustar las coordenadas de los contornos. Útil si se está trabajando con recortes de imágenes y se necesita ajustar las coordenadas de los contornos a la imagen original.

---

## **Función 5** `four_point_transform`

```python
def four_point_transform(image: MatLike, pts: List[Point])
```

### Parámetros

- **`image`**: Imágen a obtener sub-imágen.

- **`pts`**: Lista de los 4 puntos que contienen la sub-imágen deseada.

---

## **Función 6** `crop`

```python
def crop(box: tuple[float, float, float, float] | None = None)
```

### Parámetros

- **`box`**: Opcional, lista de coordenadas para hacer el recorte, el orden es `[x0, y0, x1, y1]`. Por defecto es `None`.

---

## **Función 7** `tf.keras.layers.Input`

```python
tf.keras.layers.Input(shape: tuple[int] = None, **kwargs)
```

### Parámetros

- **`shape`**: Opcional, tupla de enteros para definir la forma de las entradas. Sí no se asigna, se interpretará que no se conoce las dimensiones y puede variar.

---

## **Función 8** Data augmented

### Parámetro común

- **`fill_value`**: Valor numérico entre 0 y 255 con el que se rellenarán los límites cuando se usa `constant` como método de relleno.

- **`fill_mode`**: Forma en que se rellenan los límites según el modo dado, entre ellos:
  - **`reflect`**: Valor por defecto, refleja el borde del último píxel de la imagen para llenar los vacíos.
  - **`constant`**: Los límites se rellenan con un mismo valor (color) especificado por `fill_value`.
  - **`wrap`**: La entrada se extiende envolviendo hasta el borde opuesto.
  - **`nearest`**: La entrada se extiende por el píxel más cercano.

### `tf.keras.layers.RandomTranslation`

```python
tf.keras.layers.RandomTranslation(height_factor: float | tuple[float, float], 
                                 width_factor: float | tuple, 
                                 fill_mode: str = 'reflect', 
                                 fill_value: float = 0.0, 
                                 **kwargs)
```

#### Parámetros

- **`height_factor`**: Un número o tupla de números con valores de entre 0 y 1, que representa el límite inferior y superior para el desplazamiento vertical de manera porcentual. Un valor negativo significa desplazar la imagen hacia arriba, mientras que un valor positivo significa desplazarla hacia abajo. Si el factor es un número, este se usará para ambos límites. Ejemplo: `height_factor = .2` dará como resultado `[-20%, +20%]`.

- **`width_factor`**: Un número o tupla de números con valores de entre 0 y 1, que representa el límite inferior y superior para el desplazamiento horizontal de manera porcentual. Un valor negativo significa desplazar la imagen hacia la izquierda, mientras que un valor positivo significa desplazarla hacia la derecha.

### `tf.keras.layers.RandomRotation`

```python
tf.keras.layers.RandomRotation(factor: float | tuple[float, float], 
                              fill_mode: str = 'reflect', 
                              fill_value: float = 0.0, 
                              **kwargs)
```

#### Parámetros

- **`factor`**: Un número representado como fracción de 2π, o una tupla de tamaño 2 que representa los límites inferior y superior de la rotación en sentido horario y antihorario. Un valor positivo significa girar en el sentido contrario a las agujas del reloj, mientras que uno negativo significa girar en el sentido de las agujas del reloj.

### `tf.keras.layers.RandomZoom`

```python
tf.keras.layers.RandomZoom(height_factor: float | tuple[float, float], 
                          fill_mode: str ='reflect', 
                          fill_value: float = 0.0, 
                          **kwargs)
```

#### Parámetros

- **`height_factor`**: Un número representado como fracción del valor, o una tupla de tamaño 2 que representa el límite inferior y superior para el zoom (aumento) vertical. Cuando se representa como un único número, este valor se utiliza tanto para el límite superior como para el inferior. Un valor positivo significa alejar, mientras que un valor negativo significa acercar.

### `tf.keras.layers.RandomBrightness`

```python
tf.keras.layers.RandomBrightness(factor: float | tuple[float, float], 
                                **kwargs)
```

#### Parámetros

- **`factor`**: Número o una lista/tupla de 2 números entre -1.0 y 1.0. El factor se utiliza para determinar el límite inferior y superior del ajuste del brillo. Se elegirá aleatoriamente un valor entre los límites. Cuando se elige -1.0, la imagen de salida será negra, y cuando se elige 1.0, la imagen será totalmente blanca.

### `tf.keras.layers.RandomContrast`

```python
tf.keras.layers.RandomContrast(factor: float | tuple[float, float], 
                              **kwargs)
```

#### Parámetros

- **`factor`**: Número positivo representado como fracción del valor, o una tupla de tamaño 2 que representa los límites inferior y superior. Cuando se representa como un único número, este valor se utiliza tanto para el límite superior como para el inferior. El factor de contraste se elegirá aleatoriamente entre `[1.0 - inferior, 1.0 + superior]`. Para cualquier píxel `x` del canal, la salida será `(x - media) * factor + media`, donde `media` es el valor medio del canal.

---

## **Función 9** `tf.keras.layers.Rescaling`

```python
tf.keras.layers.Rescaling(scale: float, 
                         offset: float, 
                         **kwargs)
```

### Parámetros

- **`scale`**: La escala a aplicar a las entradas. Ejemplo: Para reescalar una entrada en el rango `[0, 255]` para que esté en el rango `[0, 1]`, pasaría `scale = 1./255` y para reescalar una entrada en el rango `[0, 255]` para que esté en el rango `[-1, 1]`, se pasaría `scale = 1./127.5` y `offset = -1`.

- **`offset`**: El desplazamiento a aplicar a las entradas.

---

## **Función 10** `tf.keras.layers.Conv2D`

```python
tf.keras.layers.Conv2D(filters: int, 
                      kernel_size: int | tuple[int, int], 
                      activation: str | any, 
                      **kwargs)
```

### Parámetros

- **`filters`**: Número de filtros. Cada filtro representa una característica que puede variar desde bordes y texturas hasta patrones más complejos; a mayor cantidad de filtros, el modelo puede aprender características más detalladas, aunque con un mayor costo computacional.

- **`kernel_size`**: Tamaño del filtro (ancho y alto), que determina el "vecindario" de la entrada que cada filtro abarca en cada paso. Sí solo se pasa un único valor, se usará el mismo para el ancho y alto. Los filtros pequeños (como 3x3) son comunes, ya que capturan bien los patrones locales y mantienen el costo computacional bajo.

- **`activation`**: Función de activación, que define cómo se manejarán los valores en cada paso. Entre las opciones, se encuentran `relu`, `leaky relu`, entre otras.

---

## **Función 11** `tf.keras.layers.MaxPool2D`

```python
tf.keras.layers.MaxPool2D(pool_size: int | tuple[int, int] = (2, 2), 
                         **kwargs)
```

### Parámetros

- **`pool_size`**: Número entero o tupla de 2 enteros, factores por los que reducir la escala `(dim1, dim2)`. Si sólo se especifica un entero, se utilizará la misma longitud de ventana para todas las dimensiones.

---

## **Función 12** `tf.keras.layers.Dense`

```python
tf.keras.layers.Dense(units: int, 
                     activation str | any = None, 
                     **kwargs)
```

### Parámetros

- **`units`**: Número entero positivo, para indicar el número de neuronas que tendrá la capa.

- **`activation`**: Función de activación.

---

## **Función 13** `tf.keras.layers.Dropout`

```python
tf.keras.layers.Dropout(rate: float, **kwargs)
```

### Parámetros

- **`rate`**: Número entre 0 y 1. Fracción de las unidades de entrada que hay que dejar caer.

---

## **Función 14** `compile`

```python
def compile(optimizer: str = 'rmsprop', 
           loss: str | any = None, 
           metrics: list[str | any] = None, 
           **kwargs)
```

### Parámetros

- **`optimizer`**: Nombre del [optimizador](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), por defecto `'rmsprop'`. El optimizador ajusta los pesos del modelo basándose en el gradiente de la función de pérdida. Existen varios optimizadores cada uno con sus ventajas y desventajas, algunos de ellos son:
  - **`RMSProp`**: Divide la tasa de aprendizaje por una media móvil del tamaño reciente de los gradientes.
  - **`AdaGrad`**: Adapta la tasa de aprendizaje según la frecuencia de actualización de cada parámetro.
  - **`Adam`**: Una combinación de AdaGrad y RMSProp, ajusta dinámicamente la tasa de aprendizaje. Es versátil y se adapta bien a la mayoría de los problemas.
  - **`SGD (Stochastic Gradient Descent)`**: Ajusta los pesos de manera incremental para cada muestra.

- **`loss`**: Nombre de la [función de perdida](https://www.tensorflow.org/api_docs/python/tf/keras/losses). Mide la diferencia entre las predicciones del modelo y los valores reales y sirve como guía para el optimizador para actualizar los pesos. Algunas de las funciones de pérdida son:
  - **`SparseCategoricalCrossentropy`**: Ideal para clasificación multiclase con etiquetas enteras.
  - **`CategoricalCrossentropy`**: Similar, pero para etiquetas one-hot. Donde las etiquetas one-hot es una forma de representar clases de manera binaria, ejemplo: `A = [1, 0, 0]`, `B = [0, 1, 0]` y `c = [0, 0, 1]`.
  - **`BinaryCrossentropy`**: Para problemas de clasificación binaria.

- **`metrics`**: Lista de [métricas](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) que debe evaluar el modelo durante el entrenamiento y las pruebas. Entre ellas están:
  - **`accuracy`**: Mide el porcentaje de predicciones correctas.
  - **`Precision, Recall, F1-Score`**: Miden la exactitud, la capacidad para identificar instancias relevantes y el balance entre precision y recall, respectivamente.

---

## **Función 15** `image_dataset_from_directory`

```python
def image_dataset_from_directory(directory: any, 
                                labels: str = 'inferred', 
                                label_mode: str = 'int', 
                                color_mode: str = 'rgb', 
                                batch_size: int = 32, 
                                image_size: tuple[int, int] = (256, 256), 
                                seed: any = None, 
                                validation_split = None, 
                                subset: str = None, 
                                pad_to_aspect_ratio: bool = False, 
                                **kwargs)
```

### Parámetros

- **`directory`**: Dirección de la carpeta que contiene los datos de entrenamiento.

- **`labels`**: Lista de etiquetas, por defecto `inferred`. Cuando es `inferred` las etiquetas se generan a partir de la estructura del directorio, cuando es `None` se indica que no hay etiquetas o simplemente le pasamos la lista de etiquetas.

- **`label_mode`**: Cadena que describe la codificación de las etiquetas. Las opciones son:
  - **`int`**: Valor por defecto, significa que las etiquetas se codifican como números enteros (por ejemplo, para `sparse_categorical_crossentropy` loss).
  - **`categorical`**: Significa que las etiquetas se codifican como un vector categórico (por ejemplo, para `categorical_crossentropy` loss).
  - **`binary`**: Significa que las etiquetas (sólo puede haber 2) se codifican como escalares float32 con valores 0 o 1 (por ejemplo, para `binary_crossentropy`).
  - **`None`**: Sin etiquetas (útil para inferencia).

- **`color_mode`**: Cadena que indica en que espacio de color obtener las imágenes, `rgb` es el valor por defecto. otros valores son `grayscale` y `rgba`.

- **`batch_size`**: Tamaño de los lotes de datos. Por defecto es 32. Si es Ninguno, los datos no se agruparán por lotes (el conjunto de datos producirá muestras individuales). Sirve para mejorar la eficiencia en memoria y procesamiento, escalabilidad, entre otras ventajas. Tener lotes muy grandes no es ideal por el uso intensivo de la memoria, además, puede generar sobreajustes.

- **`image_size`**: Tamaño al que se redimensionan las imágenes una vez leídas, especificado como `(alto, ancho)`. Por defecto es `(256, 256)`. Dado que se procesan lotes de imágenes que deben tener el mismo tamaño, debe indicarse este valor.

- **`seed`**: Semilla aleatoria opcional para barajar y transformar las imágenes. Esta semilla sirve para reproducibilidad, en caso de compartir el dataset se obtengan los mismos resultados.

- **`validation_split`**: Opcional, número entre 0 y 1. Este número indica que proporción del conjunto se tomará para datos de validación.

- **`subset`**: Cadena opcional que indica el subconjunto de datos a devolver. Sólo se utiliza si `validation_split` está activado. Los valores posibles son:
  - **`training`**: Para obtener únicamente los datos de entrenamiento.
  - **`validation`**: Para obtener únicamente los datos para validación.
  - **`both`**: Retorna una tupla con ambos conjuntos (validación y entrenamiento).

- **`pad_to_aspect_ratio`**: Si es verdadero, redimensiona las imágenes sin distorsión, rellenando las imágenes con un color negro si el tamaño de la imagen no es el que se indica.

---

## **Función 16** `fit`

```python
def fit(x: ndarray | Tensor | dict | Dataset | PyDataset = None, 
       epochs: int = 1, 
       callbacks: list[Callback] = None, 
       validation_split: float = 0.0, 
       validation_data ndarray | Tensor | dict | Dataset | PyDataset = None, 
       **kwargs)
```

### Parámetros

- **`x`**: Datos de entrada.

- **`epochs`**: Número entero que indica el número de épocas para entrenar el modelo, por defecto 1. Una época es una iteración sobre la totalidad de los datos `x`. Se debe tener en cuenta que muchas épocas no siempre significan un mejor entrenamiento.

- **`callbacks`**: Lista opcional de [Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks?hl=en" \o "https://www.tensorflow.org/api_docs/python/tf/keras/callbacks?hl=en) (funciones) a aplicar durante el entrenamiento. De entre ellas se usaron 3, las cuales son:
  - [**ModelCheckpoint**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint): Guarda el modelo en un archivo como punto de control, en algún intervalo. Con el parámetro `save_best_only = True` le indicamos que solo guarde el modelo que ha obtenido el mejor rendimiento.
  - [**EarlyStopping**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping): Detiene el entrenamiento cuando una métrica supervisada ha dejado de mejorar. Por defecto monitorea la propiedad `val_loss` por defecto, con `patience = 15` se indica el número de épocas sin mejora a esperar para detener el entrenamiento, y con `restore_best_weights = True` se indica que restaure los mejores pesos a partir de la última época con mejor valor.
  - [**CSVLogger**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger): Guarda los resultados de cada época en un archivo csv.

- **`validation_split`**: Número entre 0 y 1, por defecto 0. Este número indica que proporción del conjunto se tomará para datos de validación, estos datos no formarán parte del entrenamiento (puede usarse este parámetro para no dividir el dataset como en el paso anterior). Si se proporciona `validation_data`, este parámetro será ignorado.

- **`validation_data`**: Datos sobre los que se evalúa la pérdida y cualquier métrica del modelo al final de cada época. El modelo no se entrenará con estos datos. Por lo tanto, tenga en cuenta el hecho de que la pérdida de validación de los datos proporcionados utilizando `validation_split` o `validation_data` no se ve afectada por las capas de regularización como el ruido y el dropout.

---

> 📝 **Nota**: Este documento contiene información de referencia para funciones de procesamiento de imágenes (PIL, OpenCV) y capas de redes neuronales (TensorFlow/Keras).