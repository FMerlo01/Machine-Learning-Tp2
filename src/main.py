# src/main.py
from utils.eda_correlation import check_splits, plot_correlation_matrix

def main():
    print("=== Pipeline de Machine Learning - TP2 ===")
    plot_correlation_matrix()
    
    print("\nSiguiente paso: Entrenar clasificadores...")

if __name__ == "__main__":
    main()