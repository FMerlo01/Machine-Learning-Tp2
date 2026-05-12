import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_BASE_2D = os.path.join("results", "exp_2d")
MODEL_SLUG = "random_forest"


def _load_cv_csv(suffix):
    csv_dir = os.path.join(RESULTS_BASE_2D, MODEL_SLUG, "csv")
    csv_path = os.path.join(csv_dir, f"cv_results_{MODEL_SLUG}_{suffix}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontro {csv_path}")

    return pd.read_csv(csv_path)


def _format_label(value):
    return "None" if pd.isna(value) else value


def _plot_heatmap(df, param_x, param_y, metric_col, out_path, title):
    data = df.copy()
    data[param_x] = data[param_x].apply(_format_label)
    data[param_y] = data[param_y].apply(_format_label)

    pivot = data.pivot_table(
        index=param_y,
        columns=param_x,
        values=metric_col,
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0.8, vmax=1)
    plt.title(title)
    plt.ylabel(param_y.replace("param_", ""))
    plt.xlabel(param_x.replace("param_", ""))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rf_2d_heatmaps():
    plot_dir = os.path.join(RESULTS_BASE_2D, MODEL_SLUG, "plots")
    accuracy_dir = os.path.join(plot_dir, "accuracy")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(accuracy_dir, exist_ok=True)

    pairs = {
        "max_depth_vs_max_features": ("param_max_depth", "param_max_features"),
        "max_depth_vs_min_samples_leaf": ("param_max_depth", "param_min_samples_leaf"),
        "max_depth_vs_min_samples_split": ("param_max_depth", "param_min_samples_split"),
        "max_depth_vs_n_estimators": ("param_max_depth", "param_n_estimators"),
    }

    for suffix, (param_x, param_y) in pairs.items():
        try:
            df = _load_cv_csv(suffix)
        except FileNotFoundError:
            continue

        if param_x not in df.columns or param_y not in df.columns:
            continue

        train_out = os.path.join(
            accuracy_dir,
            f"rf_heatmap_train_accuracy_{param_x.replace('param_', '')}_vs_{param_y.replace('param_', '')}.png",
        )
        val_out = os.path.join(
            accuracy_dir,
            f"rf_heatmap_val_accuracy_{param_x.replace('param_', '')}_vs_{param_y.replace('param_', '')}.png",
        )

        _plot_heatmap(
            df,
            param_x,
            param_y,
            "mean_train_accuracy",
            train_out,
            f"RF Train Accuracy\n{param_y.replace('param_', '')} vs {param_x.replace('param_', '')}",
        )
        _plot_heatmap(
            df,
            param_x,
            param_y,
            "mean_val_accuracy",
            val_out,
            f"RF Validation Accuracy\n{param_y.replace('param_', '')} vs {param_x.replace('param_', '')}",
        )

        train_recall_out = os.path.join(
            plot_dir,
            f"rf_heatmap_train_recall_{param_x.replace('param_', '')}_vs_{param_y.replace('param_', '')}.png",
        )
        val_recall_out = os.path.join(
            plot_dir,
            f"rf_heatmap_val_recall_{param_x.replace('param_', '')}_vs_{param_y.replace('param_', '')}.png",
        )

        _plot_heatmap(
            df,
            param_x,
            param_y,
            "mean_train_recall",
            train_recall_out,
            f"RF Train Recall\n{param_y.replace('param_', '')} vs {param_x.replace('param_', '')}",
        )
        _plot_heatmap(
            df,
            param_x,
            param_y,
            "mean_val_recall",
            val_recall_out,
            f"RF Validation Recall\n{param_y.replace('param_', '')} vs {param_x.replace('param_', '')}",
        )


if __name__ == "__main__":
    plot_rf_2d_heatmaps()
