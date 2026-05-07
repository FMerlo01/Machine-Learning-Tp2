@echo off

echo   EJECUTANDO PIPELINE DE MACHINE LEARNING - TP2

echo [1/4] Creando dataset (Limpieza y Split)...
python src/data/make_dataset.py

echo.
echo [2/4] Transformando datos de Entrenamiento (Fit + Transform)...
python src/data/transform_data.py train.csv

echo.
echo [3/4] Transformando datos de Test (Only Transform)...
python src/data/transform_data.py test.csv

echo.
echo [4/4] Ejecutando Analisis y Visualizacion...
python src/main.py

echo.
echo   PROCESO FINALIZADO EXITOSAMENTE
pause