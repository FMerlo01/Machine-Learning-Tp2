import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BASE_2D = os.path.join("results", "exp_2d")
MODEL_SLUG = "random_forest"


def _load_csv(suffix):
    csv_dir = os.path.join(RESULTS_BASE_2D, MODEL_SLUG, "csv")
    csv_path = os.path.join(csv_dir, f"cv_results_{MODEL_SLUG}_{suffix}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro {csv_path}")
    return pd.read_csv(csv_path)


def _match_value(series, value):
    if value is None:
        return series.isna()
    return series == value


def _plot_bars(values, labels, ylabel, title, out_path):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4c72b0", "#dd8452"])
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0.8, 1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rf_2d_best_vs_baseline():
    plot_dir = os.path.join(RESULTS_BASE_2D, MODEL_SLUG, "plots", "best_vs_baseline")
    recall_dir = os.path.join(plot_dir, "recall")
    accuracy_dir = os.path.join(plot_dir, "accuracy")
    os.makedirs(recall_dir, exist_ok=True)
    os.makedirs(accuracy_dir, exist_ok=True)

    defaults = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    pairs = {
        "max_depth_vs_max_features": ("param_max_depth", "param_max_features"),
        "max_depth_vs_min_samples_leaf": ("param_max_depth", "param_min_samples_leaf"),
        "max_depth_vs_min_samples_split": ("param_max_depth", "param_min_samples_split"),
        "max_depth_vs_n_estimators": ("param_max_depth", "param_n_estimators"),
    }

    for suffix, (param_x, param_y) in pairs.items():
        df = _load_csv(suffix)

        if param_x not in df.columns or param_y not in df.columns:
            continue

        best_idx = df["mean_val_recall"].idxmax()
        best_row = df.loc[best_idx]

        baseline_mask = _match_value(df[param_x], defaults[param_x.replace("param_", "")])
        baseline_mask &= _match_value(df[param_y], defaults[param_y.replace("param_", "")])

        if not baseline_mask.any():
            continue

        baseline_row = df[baseline_mask].iloc[0]

        recall_values = [
            float(baseline_row["mean_val_recall"]),
            float(best_row["mean_val_recall"]),
        ]
        acc_values = [
            float(baseline_row["mean_val_accuracy"]),
            float(best_row["mean_val_accuracy"]),
        ]

        labels = ["Baseline", "Best (recall)"]
        base_title = f"RF {suffix.replace('_', ' ')}"

        recall_path = os.path.join(recall_dir, f"rf_{suffix}_baseline_vs_best_recall.png")
        acc_path = os.path.join(accuracy_dir, f"rf_{suffix}_baseline_vs_best_accuracy.png")

        _plot_bars(
            recall_values,
            labels,
            "Validation Recall",
            f"{base_title}\nRecall",
            recall_path,
        )
        _plot_bars(
            acc_values,
            labels,
            "Validation Accuracy",
            f"{base_title}\nAccuracy",
            acc_path,
        )


if __name__ == "__main__":
    plot_rf_2d_best_vs_baseline()
