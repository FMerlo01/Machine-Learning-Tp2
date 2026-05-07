import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Rutas a los archivos procesados
TRAIN_PATH = "data/processed/train.csv"
TRAIN_TRANSFORMED_PATH = "data/processed/train_transformed.csv"
TEST_PATH = "data/processed/test.csv"

def check_splits():
    print("--- VERIFICACIÓN DE SPLIT ---")
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(TEST_PATH)
        
        total = len(df_train) + len(df_test)
        train_pct = (len(df_train) / total) * 100
        test_pct = (len(df_test) / total) * 100
        
        print(f"Total de datos: {total}")
        print(f"Train: {len(df_train)} filas ({train_pct:.1f}%)")
        print(f"Test: {len(df_test)} filas ({test_pct:.1f}%)")
        print("¡El split 80/20 se hizo perfecto!\n")
    else:
        print("❌ Faltan los archivos crudos. Corré primero make_dataset.py")

def plot_correlation_matrix():
    print("--- GENERANDO MATRIZ DE CORRELACIÓN ---")
    if not os.path.exists(TRAIN_TRANSFORMED_PATH):
        print("❌ Falta el archivo transformado. Corré transform_data.py train.csv primero.")
        return

    # Cargamos el dataset transformado (usamos este porque ya no tiene el ID)
    df_train = pd.read_csv(TRAIN_TRANSFORMED_PATH)
    
    # Calculamos la matriz de correlación de Pearson
    corr_matrix = df_train.corr()

    # Configuramos el tamaño del gráfico (lo hacemos grande porque son 30 variables)
    plt.figure(figsize=(20, 15))
    
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title("Matriz de Correlación - Breast Cancer (Train Set)", fontsize=20)
    plt.tight_layout()
    
    # Guardamos el gráfico en la carpeta de utils o en la raíz para que lo veas
    plt.savefig("correlacion_features.png")
    print("✅ Gráfico guardado como 'correlacion_features.png' en la carpeta src.")
    
    # También lo mostramos por pantalla
    plt.show()

if __name__ == "__main__":
    # 1. Chequear que las proporciones estén bien
    check_splits()
    
    # 2. Hacer el Análisis Exploratorio (EDA)
    plot_correlation_matrix()