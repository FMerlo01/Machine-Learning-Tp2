import os
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
            "Modelo": f"{name}\n({params_str})",
            "Accuracy": info["best_accuracy"],
            "Recall": info["best_recall"],
        })

    df = pd.DataFrame(data)

    # Accuracy plot
    df_acc = df.sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Accuracy", y="Modelo", data=df_acc)

    plt.title("Mejor Accuracy en Validación por Modelo")
    plt.xlim(0, 1)
    plt.tight_layout()

    plt.savefig("results/best_models_accuracy.png")
    plt.close()

    # Recall plot
    df_rec = df.sort_values(by="Recall", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Recall", y="Modelo", data=df_rec)

    plt.title("Mejor Recall en Validación por Modelo")
    plt.xlim(0, 1)
    plt.tight_layout()

    plt.savefig("results/best_models_recall.png")
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
