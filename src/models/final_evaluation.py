import pandas as pd

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

TRAIN_PATH = "data/processed/train_transformed.csv"
TEST_PATH = "data/processed/test_transformed.csv"


def final_evaluation():
    print("\n=== EVALUACIÓN FINAL SOBRE TEST ===")

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Split features/targets
    X_train = train_df.drop(columns=["diagnosis"])
    y_train = train_df["diagnosis"]

    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]

    print("Modelo a Evaluar en Test: SVM, C=10, kernel linear")
    
    # Final chosen model
    model = SVC(
        C=10,
        kernel="linear",
        random_state=42,
    )

    # Train on FULL training set
    model.fit(X_train, y_train)

    # Predict on TEST
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Recall:  {recall:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
