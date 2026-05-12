import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BASE_3D = os.path.join("results", "exp_3")
MODEL_SLUG = "random_forest"


def plot_rf_exp3_bars():
    csv_dir = os.path.join(RESULTS_BASE_3D, MODEL_SLUG, "csv")
    plot_dir = os.path.join(RESULTS_BASE_3D, MODEL_SLUG, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "cv_results_random_forest_max_depth_6_7_fixed.csv")
    df = pd.read_csv(csv_path)

    if "mean_val_accuracy" not in df.columns:
        raise ValueError("No se encontro mean_val_accuracy en el CSV.")

    df = df.sort_values(by="param_max_depth")

    labels = []
    for _, row in df.iterrows():
        md = row.get("param_max_depth")
        label = "None" if pd.isna(md) else str(int(md))
        labels.append(f"max_depth={label}")

    plt.figure(figsize=(8, 5))
    plt.bar(labels, df["mean_val_accuracy"], color="#2a7f62")
    plt.ylim(0.8, 1.0)
    plt.ylabel("Validation Accuracy")
    plt.title("RF Exp3 - Accuracy por combinacion")
    plt.tight_layout()

    out_path = os.path.join(plot_dir, "rf_exp3_val_accuracy_bars.png")
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    plot_rf_exp3_bars()
