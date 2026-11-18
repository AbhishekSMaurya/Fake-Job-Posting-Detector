
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def load_data():

    X_train = load_npz("dataset/features/X_train.npz")
    X_test = load_npz("dataset/features/X_test.npz")

    y_train = np.load("dataset/features/y_train.npy")
    y_test = np.load("dataset/features/y_test.npy")

    return X_train, X_test, y_train, y_test


def get_models():

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    print("\n", "=" * 55)
    print(name)
    print("=" * 55)

    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0
    ))

    return results


def train_models():

    print("Loading feature datasets...")

    X_train, X_test, y_train, y_test = load_data()

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    models = get_models()

    results = []
    best_f1 = -1
    best_model = None
    best_model_name = None

    Path("models").mkdir(exist_ok=True)

    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(X_train, y_train)

        result = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(result)

        if result["F1 Score"] > best_f1:
            best_f1 = result["F1 Score"]
            best_model = model
            best_model_name = name

        joblib.dump(
            model,
            f"models/{name.lower().replace(' ', '_')}.pkl"
        )

    results_df = pd.DataFrame(results)

    print("\n\nMODEL COMPARISON")
    print("=" * 75)
    print(results_df.to_string(index=False))

    results_df.to_csv(
        "models/model_comparison.csv",
        index=False
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("\nBest Model:", best_model_name)
    print("Best F1 Score:", round(best_f1, 4))
    print("\nAll models saved successfully.")


if __name__ == "__main__":
    train_models()
