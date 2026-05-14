# AI-NetEval

Desarrollo de una red neuronal convolucional para obtener las respuestas de un examen y poder calificarlo. El archivo `Identificación de Respuestas.pdf` contiene toda la explicación, si les ha gustado el proyecto, dar estrellita :D.

## Instalación

### **UV**

El proyecto esta creado con el gestor de dependencias `uv`, si lo usas solo requieres ejecutar el comando:

```bash
uv sync
```

### **Vanilla**

Las dependencias también se encuentra en el archivo `requirements.txt` si es que no usas ningún gestor de dependencias. Para instalarlas, ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

## Estructura

**_Carpeta images:_** La carpeta images contiene las imágenes de entrenamiento y prueba. Cada carpeta tiene un conjunto de imágenes organizadas en subcarpetas, con el nombre de la clase a la que pertenecen.

**_Carpeta model:_** La carpeta model contiene los modelos entrenados. Cada modelo se guarda en una carpeta con el nombre de la función de activación y el sufijo augmented si se ha aplicado la transformación de datos. Además, cada carpeta tiene un archivo summary.txt con la estructura del modelo y un archivo history.json con el historial de entrenamiento.

**_Carpeta util:_** La carpeta util contiene las funciones que se utilizan en el proyecto:

- `aitest.py:` contiene funciones que sirven únicamente para mostrar imágenes y resultados mientras se entrena el modelo. No son funciones críticas para el desarrollo, pero sí para poder ir analizando los resultados obtenidos.

- `aiutil.py:` contiene funciones para cargar imágenes, transformarlas, etc. Todo lo relacionado con lo necesario para el desarrollo del proyecto.

## Uso

- `.env.local` es el archivo donde se asignan las configuraciones iniciales:

```
ACTIVATION_FUNCTION # Función de activación que usará el modelo en el entrenamiento
DATA_AUGMENTED # Indicar si se usará el aumento de datos
MODEL_ROOT_PATH # Ruta donde se guardará el modelo y sus datos
SRC_PATH # Ruta donde se encuentran las imágenes tanto de entrenamiento como de prueba
MODEL_NAME # Nombre que tendrá el modelo
IMG_SIZE # Medida estandar para los exámenes escaneados (si no se cumple se redimencionan)
IMG_CONTOUR_TARGET_SIZE # Alto mínimo que tiene un recuadro con respuestas en el exámen escaneado
```

- `dataset.py` es el archivo para crear un dataset a partir de la imagen de un examen contestado por una persona.

- `train.py` es el archivo para entrenar un modelo. Este archivo se encarga de cargar los datos, crear el modelo, compilarlo, entrenarlo y guardarlo.

- `main.py` es el archivo para utilizar un modelo ya entrenado. Este archivo se encarga de cargar el modelo, crear un dataset de prueba y hacer predicciones.

- `history.py` es el archivo para visualizar el historial de entrenamiento de un modelo. Su función es analizar el comportamiento del modelo durante el entrenamiento. Para ejecutar el código, es necesario pasar como argumento el nombre del modelo que se desea visualizar (que esté dentro de la carpeta model). Si no se le agrega ningún parámetro, usará el modelo definido en .env. Ejemplo:

```bash
uv run src/history.py # por defecto usaría el modelo definido en .env

o

python src/history.py # por defecto usaría el modelo definido en .env
```

```bash
uv run src/history.py mish

o

python src/history.py mish
```

## Créditos

Este proyecto utiliza la biblioteca [imutils](https://github.com/PyImageSearch/imutils), que está bajo la licencia MIT.
