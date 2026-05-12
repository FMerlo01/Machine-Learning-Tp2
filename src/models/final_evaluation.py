import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

TRAIN_PATH = "data/processed/train_transformed.csv"
TEST_PATH = "data/processed/test_transformed.csv"
RESULTS_DIR = "results/final_evaluation"

os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Verdadero")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def evaluate_model(model, X_train, y_train, X_test, y_test, model_desc, cm_filename):
    print(f"\n=== Evaluando: {model_desc} ===")

    # Train on FULL training set
    model.fit(X_train, y_train)

    # Predict on TEST
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall:   {recall:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, f"Matriz de Confusión:\n{model_desc}", cm_filename)

    return recall


def plot_train_val_test_comparison(recalls_dict):
    """
    recalls_dict: dict of format { "Model Name": {"Train": x, "Val": y, "Test": z} }
    """
    df = pd.DataFrame(recalls_dict).T
    
    # df has columns ["Train", "Val", "Test"] and index is "Model Name"
    ax = df.plot(kind="bar", figsize=(10, 6), color=["#4c72b0", "#dd8452", "#55a868"])
    
    plt.title("Comparación de Recall: Train vs Val vs Test", fontsize=14)
    plt.ylabel("Recall")
    plt.ylim(0, 1.15)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=10)

    plt.legend(title="Conjunto", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_val_test_recall_comparison.png"))
    plt.close()


def final_evaluation():
    print("\n=== EVALUACIÓN FINAL SOBRE TEST ===")

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Split features/targets
    X_train = train_df.drop(columns=["diagnosis"])
    y_train = train_df["diagnosis"]

    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]

    # --- Model 1: Best from train_models.py ---
    model_train = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
    )
    test_recall_1 = evaluate_model(
        model_train, 
        X_train, y_train, X_test, y_test, 
        "Mejor Modelo Baseline (RF n_est=150)",
        "cm_baseline_rf.png"
    )

    # Metrics from CV results (from results/cv_results_random_forest.csv)
    cv_train_recall_1 = 1.0000
    cv_val_recall_1 = 0.9588

    # --- Model 2: Best from exp_2d ---
    model_exp2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.7,
        bootstrap=True,
        criterion="gini",
        random_state=42,
    )
    test_recall_2 = evaluate_model(
        model_exp2, 
        X_train, y_train, X_test, y_test, 
        "Mejor Modelo Exp 2D (RF max_depth=7, max_feat=0.7)",
        "cm_exp2_rf.png"
    )

    # Metrics from CV results (from results/exp_2d/.../cv_results_random_forest_max_depth_vs_max_features.csv)
    cv_train_recall_2 = 1.0000
    cv_val_recall_2 = 0.9647

    # --- Comparación Gráfica ---
    recalls_dict = {
        "Baseline RF\n(n_est=150)": {
            "Train": cv_train_recall_1,
            "Val": cv_val_recall_1,
            "Test": test_recall_1
        },
        "Exp 2D RF\n(max_depth=7, max_features=0.7)": {
            "Train": cv_train_recall_2,
            "Val": cv_val_recall_2,
            "Test": test_recall_2
        }
    }
    plot_train_val_test_comparison(recalls_dict)
    
    print(f"\n✅ Gráficos guardados en el directorio: {RESULTS_DIR}")

if __name__ == "__main__":
    final_evaluation()