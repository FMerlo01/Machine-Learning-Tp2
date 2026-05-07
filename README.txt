PROYECTO DE CLASIFICACION SUPERVISADA - BREAST CANCER WISCONSIN

DESCRIPCION DEL PROYECTO
Este proyecto implementa un pipeline completo de Machine Learning para predecir si un tumor de seno es maligno o benigno utilizando el dataset de Breast Cancer Wisconsin. Se enfoca en la limpieza de datos por multicolinealidad, optimizacion de hiperparametros y evaluacion de multiples modelos.

ESTRUCTURA DEL PROYECTO

    data/raw/: Contiene el dataset original.

    data/processed/: Contiene los datos limpios y escalados tras el procesamiento.

    src/data/: Scripts para limpieza (make_dataset.py) y escalado (transform_data.py).

    src/models/: Logica de entrenamiento y GridSearchCV (train_models.py).

    src/main.py/: Orquestador principal que ejecuta el entrenamiento y genera graficos.

    results/: Graficos de comparacion y analisis de resultados.

PASOS DE EJECUCION


    Toda la ejecucion se realiza automatica y secuencialmente corriendo en terminal el script correspondiente a tu SO: "run_pipeline.bat" (Windows) o "run_pipeline.sh" (Linux)
    Se recomienda ingresar primero a un visual enviorment propio e instalar las dependencias del proyecto con "pip install -r requierements.txt"

    Preparacion y Limpieza:
    Se ejecuta 'python src/data/make_dataset.py'. Este paso realiza dos limpiezas criticas:

    Limpieza 1: Elimina variables de tamaño redundantes (Perimeter, Area).

    Limpieza 2: Elimina variables de forma correlacionadas (Compactness, Concavity), manteniendo Concave Points por su alta correlacion con el diagnostico.

    Transformacion:
    Se ejecuta 'python src/data/transform_data.py' para aplicar StandardScaler a los conjuntos de train y test por separado.

    Entrenamiento y Evaluacion:
    Se ejecuta 'python src/main.py'. El script realiza:

    Validacion Cruzada (K-Fold, K=5) sobre el set de entrenamiento.

    Busqueda de hiperparametros (GridSearchCV) para SVM, Random Forest y KNN.

    Generacion de reportes en consola y graficos de comparacion en la carpeta results/.

MODELOS EVALUADOS

    Support Vector Machine (SVM) - Lider en performance con kernel lineal.

    Random Forest Classifier.

    K-Nearest Neighbors (KNN).

    Gaussian Naive-Bayes.

    Linear Discriminant Analysis (LDA).

REQUISITOS

    Python 3.8 o superior.

    Las librerias especificas se encuentras listadas en "requirements.txt"