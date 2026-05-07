# src/main.py
from asyncio import subprocess
import os
from utils.eda_correlation import check_splits, plot_correlation_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda_correlation import check_splits
from models.train_models import run_training

def plot_best_with_params(results):

    os.makedirs("results", exist_ok=True)
    
    print("\n--- GRÁFICO 1: MEJORES MODELOS CON PARÁMETROS ---")
    data = []
    for name, info in results.items():
        # Formateamos los parámetros para que no ocupen tanto espacio
        params_str = str(info['best_params']).replace('{', '').replace('}', '').replace("'", "")
        data.append({'Modelo': f"{name}\n({params_str})", 'Accuracy': info['best_score']})
    
    df = pd.DataFrame(data).sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Accuracy', y='Modelo', data=df, hue='Modelo', palette='magma', legend=False)
    plt.xlim(0.85, 1.0)
    plt.title('Mejor Accuracy por Modelo (Hyperparameters en etiquetas)', fontsize=14)
    
    # Anotaciones de texto
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}", (p.get_width(), p.get_y() + p.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', va='center')
    
    plt.tight_layout()
    plt.savefig("results/mejor_modelo_params.png")
    plt.show()

def plot_all_combinations(results):
    print("\n--- GRÁFICO 2: TODAS LAS COMBINACIONES DEL GRID ---")
    all_data = []
    
    print(f"{'Modelo':<15} | {'Accuracy':<10} | {'Parámetros'}")
    print("-" * 60)

    for name, info in results.items():
        for score, params in zip(info['all_scores'], info['all_params']):
            param_label = str(params).replace('{', '').replace('}', '').replace("'", "")
            all_data.append({'Modelo': name, 'Accuracy': score, 'Params': param_label})
            print(f"{name:<15} | {score:.4f}     | {param_label}")
    
    df_all = pd.DataFrame(all_data)

    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Modelo', y='Accuracy', data=df_all, hue='Modelo', 
                  jitter=True, size=9, alpha=0.6, palette='deep', legend=False)
    
    plt.title('Distribución de Accuracy de todas las combinaciones', fontsize=14)
    plt.ylabel('Mean CV Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("results/todas_combinaciones.png")
    plt.show()

def main():
    print("=== Pipeline de Machine Learning - TP2 ===")
    #plot_correlation_matrix()

    # 2. Entrenar y obtener resultados
    dict_results = run_training()
    
    # 3. Graficar comparación
    plot_best_with_params(dict_results)
    plot_all_combinations(dict_results)

if __name__ == "__main__":
    main()