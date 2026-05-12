import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BASE_1D = os.path.join("results", "exp_1d")
MODEL_SLUG = "random_forest"


def _load_cv_csv(suffix):
    csv_dir = os.path.join(RESULTS_BASE_1D, MODEL_SLUG, "csv")
    csv_path = os.path.join(csv_dir, f"cv_results_{MODEL_SLUG}_{suffix}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro {csv_path}")

    return pd.read_csv(csv_path)


def _sorted_param_labels(series):
    raw_vals = series.tolist()
    numeric_vals = [v for v in raw_vals if pd.notna(v)]
    has_nan = any(pd.isna(v) for v in raw_vals)

    if numeric_vals and all(isinstance(v, (int, float)) for v in numeric_vals):
        ordered = sorted(set(numeric_vals))
        labels = [str(v) for v in ordered]
        if has_nan:
            ordered.append(None)
            labels.append("None")
        return ordered, labels

    unique_vals = []
    for v in raw_vals:
        label = "None" if pd.isna(v) else str(v)
        if label not in unique_vals:
            unique_vals.append(label)
    return unique_vals, unique_vals


def _plot_param_curve(df, param_col, metric, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    values, labels = _sorted_param_labels(df[param_col])
    data = df.copy()

    if all(isinstance(v, (int, float)) for v in values if v is not None):
        data = data.sort_values(by=param_col)
        x_vals = data[param_col].tolist()
        x_labels = ["None" if pd.isna(v) else str(v) for v in x_vals]
        x_positions = list(range(len(x_vals)))
    else:
        data[param_col] = data[param_col].apply(lambda v: "None" if pd.isna(v) else str(v))
        data = data.set_index(param_col).reindex(labels).reset_index()
        x_positions = list(range(len(labels)))
        x_labels = labels

    plt.figure(figsize=(9, 6))

    if metric == "accuracy":
        plt.plot(
            x_positions,
            data["mean_train_accuracy"],
            marker="o",
            label="Train Accuracy",
        )
        plt.plot(
            x_positions,
            data["mean_val_accuracy"],
            marker="o",
            label="Validation Accuracy",
        )
        title = "Accuracy"
    else:
        plt.plot(
            x_positions,
            data["mean_train_recall"],
            marker="o",
            label="Train Recall",
        )
        plt.plot(
            x_positions,
            data["mean_val_recall"],
            marker="o",
            label="Validation Recall",
        )
        title = "Recall"

    plt.xticks(x_positions, x_labels)
    plt.xlabel(param_col.replace("param_", ""))
    plt.ylabel("Score")
    plt.title(f"Random Forest - {title} vs {param_col.replace('param_', '')}")
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_name = f"rf_curve_{param_col.replace('param_', '')}_{metric}.png"
    plt.savefig(os.path.join(plot_dir, out_name))
    plt.close()


def plot_rf_1d_curves():
    plot_dir = os.path.join(RESULTS_BASE_1D, MODEL_SLUG, "plots")

    sweep_files = {
        "param_max_depth": "max_depth",
        "param_n_estimators": "n_estimators",
        "param_min_samples_split": "min_samples_split",
        "param_min_samples_leaf": "min_samples_leaf",
        "param_max_features": "max_features",
    }

    for param_col, suffix in sweep_files.items():
        try:
            df = _load_cv_csv(suffix)
        except FileNotFoundError:
            continue

        if param_col not in df.columns:
            continue

        _plot_param_curve(df, param_col, "accuracy", plot_dir)
        _plot_param_curve(df, param_col, "recall", plot_dir)


if __name__ == "__main__":
    plot_rf_1d_curves()
