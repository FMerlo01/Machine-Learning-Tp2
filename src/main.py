# src/main.py
from utils.eda_correlation import check_splits, plot_correlation_matrix

def main():
    print("=== Pipeline de Machine Learning - TP2 ===")
    
    # 1. Verificar que el split se hizo correctamente
    check_splits()
    
    # 2. Generar el Análisis Exploratorio de Datos (EDA)
    # Esto te ayudará a justificar en tu presentación qué variables están superpuestas
    plot_correlation_matrix()
    
    print("\nSiguiente paso: Entrenar clasificadores...")

if __name__ == "__main__":
    main()