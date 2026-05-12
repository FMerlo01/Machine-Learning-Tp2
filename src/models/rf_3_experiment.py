import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TRAIN_TRANSFORMED_PATH = "data/processed/train_transformed.csv"
RESULTS_BASE_3D = os.path.join("results", "exp_3")
MODEL_NAME = "Random Forest"
MODEL_SLUG = "random_forest"


def _save_cv_results_csv(grid_search, suffix):
    csv_dir = os.path.join(RESULTS_BASE_3D, MODEL_SLUG, "csv")
    os.makedirs(csv_dir, exist_ok=True)

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
    out_df.insert(0, "model", MODEL_NAME)

    out_path = os.path.join(csv_dir, f"cv_results_{MODEL_SLUG}_{suffix}.csv")
    out_df.to_csv(out_path, index=False)


def run_rf_exp3():
    print("=== RF EXPERIMENTO 3 (CV) ===")

    df_train = pd.read_csv(TRAIN_TRANSFORMED_PATH)
    X_train = df_train.drop(columns=["diagnosis"])
    y_train = df_train["diagnosis"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        criterion="gini",
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
    }

    param_grid = {
        "max_depth": [6, 7],
        "max_features": [0.7],
        "min_samples_leaf": [2],
        "min_samples_split": [4],
        "n_estimators": [150],
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=scoring,
        refit="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    _save_cv_results_csv(grid_search, suffix="max_depth_6_7_fixed")
    print("✅ RF exp3 procesado: max_depth 6/7 con parametros fijos.")


if __name__ == "__main__":
    run_rf_exp3()
