
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import save_npz

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


TRAIN_PATH = "dataset/processed/train.csv"
TEST_PATH = "dataset/processed/test.csv"

TEXT_COLUMN = "combined_text"

CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

NUMERICAL_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_range_missing",
    "department_missing",
    "company_profile_missing",
    "benefits_missing",
    "requirements_missing",
    "industry_missing",
    "required_education_missing",
    "required_experience_missing"
]


def build_feature_engineer():

    text_pipeline = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True
    )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
            )
        )
    ])

    feature_engineer = ColumnTransformer(
        transformers=[
            (
                "text",
                text_pipeline,
                TEXT_COLUMN
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_COLUMNS
            )
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    return feature_engineer


def run_feature_engineering():

    print("Loading processed datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["fraudulent"])
    y_train = train_df["fraudulent"]

    X_test = test_df.drop(columns=["fraudulent"])
    y_test = test_df["fraudulent"]

    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))

    print("\nBuilding Feature Engineering Pipeline...")

    feature_engineer = build_feature_engineer()

    print("\nFitting features on training data...")

    X_train_features = feature_engineer.fit_transform(X_train)

    print("Transforming testing data...")

    X_test_features = feature_engineer.transform(X_test)

    output_dir = Path("dataset/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_dir / "X_train.npz",
        X_train_features
    )

    save_npz(
        output_dir / "X_test.npz",
        X_test_features
    )

    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    joblib.dump(
        feature_engineer,
        "models/feature_engineer.pkl"
    )

    print("\nFEATURE ENGINEERING COMPLETED")
    print("-" * 45)

    print("Training Feature Shape:", X_train_features.shape)
    print("Testing Feature Shape:", X_test_features.shape)

    print("\nTraining Matrix Format:", X_train_features.format)
    print("Testing Matrix Format:", X_test_features.format)

    print("\nTraining Matrix Non-zero Values:",
          X_train_features.nnz)

    print("Testing Matrix Non-zero Values:",
          X_test_features.nnz)

    print("\nSaved files:")
    print("dataset/features/X_train.npz")
    print("dataset/features/X_test.npz")
    print("dataset/features/y_train.npy")
    print("dataset/features/y_test.npy")
    print("models/feature_engineer.pkl")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    run_feature_engineering()
