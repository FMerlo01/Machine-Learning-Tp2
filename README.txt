TP2: Clasificación Supervisada - Breast Cancer Wisconsin
Proyecto de Machine Learning (72.75) para predecir si un tumor de seno es benigno o maligno utilizando características extraídas de imágenes digitales de punciones (FNA).

Requisitos Previos

    Python instalado (versión 3.8 o superior recomendada).

    Instala las dependencias del proyecto ejecutando el siguiente comando en la raíz: pip install -r requirements.txt

Instrucciones de Ejecución

1. Preparar el Dataset
Este script toma los datos crudos (data/raw/breast_cancer.csv), elimina el ID, codifica el diagnóstico (Maligno = 1, Benigno = 0) y realiza la división 80/20 de forma estratificada para mantener las proporciones de clases.
Comando: 
python src/data/make_dataset.py

(Generará train.csv y test.csv en data/processed/)

2. Transformar los Datos
Aplica escalado (StandardScaler) a las variables continuas. Es vital correr primero el entrenamiento para aprender los parámetros del escalador sin cometer data leakage.
Comandos:

python src/data/transform_data.py train.csv
python src/data/transform_data.py test.csv

(Generará train_transformed.csv y test_transformed.csv, además de guardar el escalador en data/processed/scaler.pkl)

3. Entrenar y Evaluar los Modelos
Para ejecutar el pipeline principal que busca los mejores hiperparámetros usando GridSearchCV y evalúa los modelos:
Comandos:

cd src
python main.py