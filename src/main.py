# src/main.py
from asyncio import subprocess
import os
from utils.eda_correlation import check_splits, plot_correlation_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda_correlation import check_splits
from models.train_models import run_training
from models.train_models import run_training
from plots.model_comparison import (
    plot_best_with_params,
    plot_all_combinations,
)
from plots.validation_curves import (
    plot_validation_curves,
)

def main():
    print("=== Pipeline de Machine Learning - TP2 ===")
    #plot_correlation_matrix()

    # 2. Entrenar y obtener resultados
    dict_results = run_training()
    
    # 3. Graficar comparación
    plot_best_with_params(dict_results)
    plot_all_combinations(dict_results)
    plot_validation_curves(dict_results)
    print("Fotos guardadas en results/")

if __name__ == "__main__":
    main()
