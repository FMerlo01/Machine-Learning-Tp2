import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TRAIN_TRANSFORMED_PATH = "data/processed/train_transformed.csv"
RESULTS_BASE_2D = os.path.join("results", "exp_2d")
MODEL_NAME = "Random Forest"
MODEL_SLUG = "random_forest"


def _save_cv_results_csv(grid_search, suffix):
    csv_dir = os.path.join(RESULTS_BASE_2D, MODEL_SLUG, "csv")
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


def run_rf_2d_experiments():
    print("=== RF 2D EXPERIMENTOS (CV) ===")

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

    rf_sweeps = {
        "max_depth": list(range(1, 16)) + [None],
        "n_estimators": list(range(50, 501, 50)),
        "min_samples_split": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100],
        "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100],
        "max_features": ["sqrt", "log2", None, 0.5, 0.7],
    }

    rf_pair_sweeps = {
        "max_depth_vs_max_features": {
            "max_depth": rf_sweeps["max_depth"],
            "max_features": rf_sweeps["max_features"],
        },
        "max_depth_vs_min_samples_leaf": {
            "max_depth": rf_sweeps["max_depth"],
            "min_samples_leaf": rf_sweeps["min_samples_leaf"],
        },
        "max_depth_vs_min_samples_split": {
            "max_depth": rf_sweeps["max_depth"],
            "min_samples_split": rf_sweeps["min_samples_split"],
        },
        "max_depth_vs_n_estimators": {
            "max_depth": rf_sweeps["max_depth"],
            "n_estimators": rf_sweeps["n_estimators"],
        },
    }

    for pair_name, pair_grid in rf_pair_sweeps.items():
        grid_search = GridSearchCV(
            model,
            pair_grid,
            cv=5,
            scoring=scoring,
            refit="recall",
            return_train_score=True,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        _save_cv_results_csv(grid_search, suffix=pair_name)
        print(f"✅ RF 2D procesado: {pair_name}.")


if __name__ == "__main__":
    run_rf_2d_experiments()
