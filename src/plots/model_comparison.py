import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_best_with_params(results):
    os.makedirs("results", exist_ok=True)

    data = []

    for name, info in results.items():
        params_str = ", ".join(
            f"{k}={v}" for k, v in info["best_params"].items()
        )

        data.append({
            "Modelo": name,
            "Accuracy": info["best_accuracy"],
            "Recall": info["best_recall"],
            "Train Recall": info["best_train_recall"],
            "Params": params_str,
        })

    df = pd.DataFrame(data)

    # Accuracy plot
    df_acc = df.sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Modelo", y="Accuracy", data=df_acc)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    plt.title("Mejor Accuracy en Validación por Modelo")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("results/best_models_accuracy.png")
    plt.close()

    # Recall plot
    df_rec = df.sort_values(by="Recall", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Modelo", y="Recall", data=df_rec)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    plt.title("Mejor Recall en Validación por Modelo")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("results/best_models_recall.png")
    plt.close()

    # Train vs validation recall
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.4
    plt.bar(x - width / 2, df["Recall"], width, label="Val Recall")
    plt.bar(x + width / 2, df["Train Recall"], width, label="Train Recall")
    plt.xticks(x, df["Modelo"], rotation=30)
    plt.ylim(0, 1)
    plt.title("Recall Train vs Validacion por Modelo")
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/best_models_recall_train_val.png")
    plt.close()

    # Params summary
    params_df = df[["Modelo", "Params"]].copy()
    params_df = params_df.sort_values(by="Modelo")

    plt.figure(figsize=(12, 0.6 * len(params_df) + 2))
    plt.axis("off")
    plt.title("Mejores hiperparametros por modelo")
    plt.table(
        cellText=params_df.values,
        colLabels=params_df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    plt.tight_layout()

    plt.savefig("results/best_models_params.png")
    plt.close()


def plot_best_with_params_from_csv(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(results_dir, "cv_results_*.csv")))
    if not csv_paths:
        raise FileNotFoundError("No se encontraron CSVs de cv_results en results/.")

    data = []

    for path in csv_paths:
        df = pd.read_csv(path)
        if "mean_val_recall" not in df.columns:
            continue

        model_name = str(df["model"].iloc[0]) if "model" in df.columns else os.path.basename(path)
        idx = df["mean_val_recall"].idxmax()
        row = df.loc[idx]

        params = {
            col.replace("param_", ""): row[col]
            for col in df.columns
            if col.startswith("param_")
        }
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        data.append({
            "Modelo": model_name,
            "Accuracy": float(row["mean_val_accuracy"]),
            "Recall": float(row["mean_val_recall"]),
            "Train Recall": float(row["mean_train_recall"]),
            "Params": params_str,
        })

    if not data:
        raise ValueError("No se encontraron columnas de metricas en los CSVs.")

    df = pd.DataFrame(data)

    # Accuracy plot
    df_acc = df.sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Modelo", y="Accuracy", data=df_acc)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    plt.title("Mejor Accuracy en Validacion por Modelo")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "best_models_accuracy.png"))
    plt.close()

    # Recall plot
    df_rec = df.sort_values(by="Recall", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Modelo", y="Recall", data=df_rec)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    plt.title("Mejor Recall en Validacion por Modelo")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "best_models_recall.png"))
    plt.close()

    # Train vs validation recall
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.4
    plt.bar(x - width / 2, df["Recall"], width, label="Val Recall")
    plt.bar(x + width / 2, df["Train Recall"], width, label="Train Recall")
    plt.xticks(x, df["Modelo"], rotation=30)
    plt.ylim(0, 1)
    plt.title("Recall Train vs Validacion por Modelo")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "best_models_recall_train_val.png"))
    plt.close()

    # Params summary
    params_df = df[["Modelo", "Params"]].copy()
    params_df = params_df.sort_values(by="Modelo")

    plt.figure(figsize=(12, 0.6 * len(params_df) + 2))
    plt.axis("off")
    plt.title("Mejores hiperparametros por modelo")
    plt.table(
        cellText=params_df.values,
        colLabels=params_df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "best_models_params.png"))
    plt.close()


def plot_all_combinations(results):
    os.makedirs("results", exist_ok=True)

    all_data = []

    for name, info in results.items():
        for acc, rec, params in zip(
            info["mean_val_accuracy"],
            info["mean_val_recall"],
            info["all_params"],
        ):
            param_label = ", ".join(
                f"{k}={v}" for k, v in params.items()
            )

            all_data.append({
                "Modelo": name,
                "Accuracy": acc,
                "Recall": rec,
                "Params": param_label,
            })

    df = pd.DataFrame(all_data)

    # Accuracy
    plt.figure(figsize=(12, 6))

    sns.scatterplot(
        data=df,
        x="Modelo",
        y="Accuracy",
        hue="Modelo",
        legend=False,
    )

    plt.title("Accuracy en Validación para todas las combinaciones de hiperparámetros")
    plt.ylim(0.8, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("results/all_combinations_accuracy.png")
    plt.close()

    # Recall
    plt.figure(figsize=(12, 6))

    sns.scatterplot(
        data=df,
        x="Modelo",
        y="Recall",
        hue="Modelo",
        legend=False,
    )

    plt.title("Recall en Validación para todas las combinaciones de hiperparámetros")
    plt.ylim(0.8, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("results/all_combinations_recall.png")
    plt.close()
