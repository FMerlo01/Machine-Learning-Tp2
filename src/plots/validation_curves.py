import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_validation_curves(results):
    os.makedirs("results", exist_ok=True)

    for model_name, info in results.items():
        df = pd.DataFrame(info["all_params"])

        df["train_accuracy"] = info["mean_train_accuracy"]
        df["val_accuracy"] = info["mean_val_accuracy"]

        df["train_recall"] = info["mean_train_recall"]
        df["val_recall"] = info["mean_val_recall"]

        if model_name == "SVM":
            param = "C"

            if "kernel" in df.columns:
                df = df[df["kernel"] == "rbf"]

        elif model_name == "KNN":
            param = "n_neighbors"

            if "weights" in df.columns:
                df = df[df["weights"] == "uniform"]

        elif model_name == "Random Forest":
            param = "max_depth"

            if "n_estimators" in df.columns:
                df = df[df["n_estimators"] == 100]

        else:
            continue

        df = df.dropna(subset=[param])
        df = df.sort_values(by=param)

        if df.empty:
            continue

        plt.figure(figsize=(9, 6))

        plt.plot(
            df[param],
            df["train_accuracy"],
            marker="o",
            label="Train Accuracy",
        )

        plt.plot(
            df[param],
            df["val_accuracy"],
            marker="o",
            label="Validation Accuracy",
        )

        plt.plot(
            df[param],
            df["train_recall"],
            marker="o",
            label="Train Recall",
        )

        plt.plot(
            df[param],
            df["val_recall"],
            marker="o",
            label="Validation Recall",
        )

        plt.xlabel(param)
        plt.ylabel("Score")

        plt.title(f"Curva de Validación de {model_name}")

        plt.ylim(0.8, 1.05)

        plt.legend()

        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(f"results/{filename}_validation_curve.png")

        plt.close()
