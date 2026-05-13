import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

TRAIN_TRANSFORMED_PATH = "data/processed/train_transformed.csv"
RESULTS_DIR = "results"


def _model_slug(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def _save_cv_results_csv(model_name, grid_search):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cv_df = pd.DataFrame(grid_search.cv_results_)

    metric_cols = [
        "mean_train_accuracy",
        "mean_test_accuracy",
        "mean_train_recall",
        "mean_test_recall",
    ]

    param_cols = [col for col in cv_df.columns if col.startswith("param_")]

    out_df = cv_df[metric_cols + param_cols].copy()
    out_df = out_df.rename(
        columns={
            "mean_test_accuracy": "mean_val_accuracy",
            "mean_test_recall": "mean_val_recall",
        }
    )
    out_df.insert(0, "model", model_name)

    out_path = os.path.join(
        RESULTS_DIR,
        f"cv_results_{_model_slug(model_name)}.csv",
    )
    out_df.to_csv(out_path, index=False)

def run_training():
    print("=== ENTRENAMIENTO Y EVALUACIÓN DE MODELOS (CV) ===")
    
    # 1. Cargar datos de entrenamiento escalados
    df_train = pd.read_csv(TRAIN_TRANSFORMED_PATH)
    X_train = df_train.drop(columns=['diagnosis'])
    y_train = df_train['diagnosis']
    
    # 2. Definir los modelos base
    models = {
        "Naive-Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    # 3. Definir los hiperparámetros a explorar (Punto 3.1 del TP)
    param_grids = {
        "Naive-Bayes": {}, # NB no tiene hiperparámetros críticos para ajustar acá
        "LDA": {},         # Idem LDA
        "SVM": {
            "C": [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 15, 20],
            "kernel": ["linear", "rbf"],
        },
        "KNN": {
            "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "Random Forest": {
            "n_estimators": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
            "max_depth": [None, 3, 5, 7, 9, 11, 15, 20],
            "min_samples_split": [2, 3, 5, 7, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
    }
    
    results_detail = {}

    for name, model in models.items():
        scoring = {
            "accuracy": "accuracy",
            "recall": "recall",
        }

        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=5,
            scoring=scoring,
            refit="recall",
            return_train_score=True,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        _save_cv_results_csv(name, grid_search)
        
        # Guardamos TODO: el mejor score, los parámetros y TODOS los resultados del grid
        results_detail[name] = {
            "best_accuracy": grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_],
            "best_recall": grid_search.cv_results_["mean_test_recall"][grid_search.best_index_],
            "best_train_recall": grid_search.cv_results_["mean_train_recall"][grid_search.best_index_],
            "best_params": grid_search.best_params_,

            "mean_train_accuracy": grid_search.cv_results_["mean_train_accuracy"],
            "mean_val_accuracy": grid_search.cv_results_["mean_test_accuracy"],

            "mean_train_recall": grid_search.cv_results_["mean_train_recall"],
            "mean_val_recall": grid_search.cv_results_["mean_test_recall"],

            "all_params": grid_search.cv_results_["params"],
        }
        print(f"✅ {name} procesado.")

    return results_detail


if __name__ == "__main__":
    run_training()
