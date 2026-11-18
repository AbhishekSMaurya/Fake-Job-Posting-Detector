
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def analyze_thresholds():

    X_train = load_npz("dataset/features/X_train.npz")
    y_train = np.load("dataset/features/y_train.npy")

    X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    print("Training Logistic Regression on training subset...")

    model.fit(X_subtrain, y_subtrain)

    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_validation, predictions, zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    print("\nTHRESHOLD COMPARISON")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nCONFUSION MATRICES")

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        print(f"\nThreshold: {threshold}")
        print(confusion_matrix(y_validation, predictions))


if __name__ == "__main__":
    analyze_thresholds()
