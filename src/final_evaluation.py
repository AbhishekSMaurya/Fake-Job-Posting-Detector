
import numpy as np
import joblib

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def final_evaluation():

    X_train = load_npz("dataset/features/X_train.npz")
    X_test = load_npz("dataset/features/X_test.npz")

    y_train = np.load("dataset/features/y_train.npy")
    y_test = np.load("dataset/features/y_test.npy")

    print("Training final Logistic Regression model...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    threshold = 0.50

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("\nFINAL MODEL PERFORMANCE")
    print("=" * 55)

    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nAdditional Metrics")
    print("-" * 40)
    print("Threshold:", threshold)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions), 4))
    print("Recall:", round(recall_score(y_test, predictions), 4))
    print("F1 Score:", round(f1_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    joblib.dump(model, "models/final_model.pkl")

    print("\nFinal model saved at models/final_model.pkl")


if __name__ == "__main__":
    final_evaluation()
