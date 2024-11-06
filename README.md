# AI-NetEval

Desarrollo de una red neuronal convolucional para obtener las respuestas de un examen y poder calificarlo. El archivo ```Identificación de Respuestas.pdf``` contiene toda la explicación, si les ha gustado el proyecto, dar estrellita :D.

## Instalación

Las dependencias se encuentran en el archivo `requirements.txt`. Para instalarlas, ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

## Estructura

***Carpeta images:*** La carpeta images contiene las imágenes de entrenamiento y prueba. Cada carpeta tiene un conjunto de imágenes organizadas en subcarpetas, con el nombre de la clase a la que pertenecen.

***Carpeta model:*** La carpeta model contiene los modelos entrenados. Cada modelo se guarda en una carpeta con el nombre de la función de activación y el sufijo augmented si se ha aplicado la transformación de datos. Además, cada carpeta tiene un archivo summary.txt con la estructura del modelo y un archivo history.json con el historial de entrenamiento.

***Carpeta util:*** La carpeta util contiene las funciones que se utilizan en el proyecto:

- ```aitest.py:``` contiene funciones que sirven únicamente para mostrar imágenes y resultados mientras se entrena el modelo. No son funciones críticas para el desarrollo, pero sí para poder ir analizando los resultados obtenidos.

- ```aiutil.py:``` contiene funciones para cargar imágenes, transformarlas, etc. Todo lo relacionado con lo necesario para el desarrollo del proyecto.

## Uso

- ```dataset.py``` es el archivo para crear un dataset a partir de la imagen de un examen contestado por una persona.

- ```train.py``` es el archivo para entrenar un modelo. Este archivo se encarga de cargar los datos, crear el modelo, compilarlo, entrenarlo y guardarlo.

- ```use.py``` es el archivo para utilizar un modelo ya entrenado. Este archivo se encarga de cargar el modelo, crear un dataset de prueba y hacer predicciones.

- ```history.py``` es el archivo para visualizar el historial de entrenamiento de un modelo. Su función es analizar el comportamiento del modelo durante el entrenamiento. Para ejecutar el código, es necesario pasar como argumento el nombre del modelo que se desea visualizar (que esté dentro de la carpeta model). Si no se le agrega ningún parámetro, usará mish_augmented. Ejemplo:

```bash
python history.py # por defecto usaría mish_augmented
```

```bash
python history.py mish
```

## Créditos

Este proyecto utiliza la biblioteca [imutils](https://github.com/PyImageSearch/imutils), que está bajo la licencia MIT.
