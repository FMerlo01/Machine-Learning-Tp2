echo   EJECUTANDO PIPELINE DE MACHINE LEARNING - TP2

echo [1/4] Creando dataset
python3 src/data/make_dataset.py

echo "[2/4] Transformando datos de Entrenamiento (Fit + Transform)..."
python3 src/data/transform_data.py train.csv

echo "[3/4] Transformando datos de Test (Only Transform)..."
python3 src/data/transform_data.py test.csv

echo "[4/4] Ejecutando Analisis y Visualizacion..."
python3 src/main.py


