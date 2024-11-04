# AI-NetEval

Desarrollo de red neuronal convolucional para obtener las respuestas de un examen y poder calificarlo.

## Instalación

Las dependencias se encuentran en el archivo `requirements.txt`, para instalrlas hay que ejecutar el comando:

```python
pip install -r requirements.txt
```

## Estructura

### Carpeta _images_

La carpeta _images_ contiene las imágenes de entrenamiento y prueba. Cada carpeta tiene un conjunto de imágenes organizadas en subcarpetas con el nombre de la clase a la que pertenecen.

### Carpeta _model_

La carpeta _model_ contiene los modelos entrenados. Cada modelo se guarda en una carpeta con el nombre de la función de activación y el sufijo _augmented_ si se ha aplicado la transformación de datos, además, cada carpeta tiene un archivo _summary.txt_ con la estructura del modelo y un archivo _history.json_ con el historial de entrenamiento.

### Carpeta _util_

La carpeta _util_ contiene las funciones que se utilizan en el proyecto.:

`aitest.py` contiene funciones que sirven unicamente para mostrar imágenes y resultados mientras se entrena el modelo, no son funciones críticas para el desarrollo pero si para poder ir analizando los resultados obtenidos.

`aiutil.py` contiene funciones para cargar imágenes, transformarlas, etc, todo lo relacionado con lo necesario para el desarrollo del proyecto.

## Uso

`dataset.py` es el archivo para poder crear un dataset a partir de la imágen de un exámen contestado por una persona.

`train.py` es el archivo para poder entrenar un modelo. Este archivo se encarga de cargar los datos, crear el modelo, compilarlo, entrenarlo y guardarlo.

`use.py` es el archivo para poder utilizar un modelo ya entrenado. Este archivo se encarga de cargar el modelo, crear un dataset de prueba y hacer predicciones.

`history.py` es el archivo para poder visualizar el historial de entrenamiento de un modelo. Su función es poder analisar el comportamiento del modelo durante el entrenamiento. Para ejecutar el código, es necesario pasarle como argumento el nombre del modelo que se desea visualizar (que esten dentro de la carpeta model), si no se le agrega ningún parámetro, usará mish_augmented. Ejemplo:

```python
python history.py mish # por defecto mish_augmented
```

## Créditos

Este proyecto utiliza la biblioteca [imutils](https://github.com/PyImageSearch/imutils) que está bajo la licencia MIT. 
