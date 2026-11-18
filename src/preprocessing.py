
import pandas as pd
import re
import html
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = "dataset/fake_job_postings.csv"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

MISSING_INDICATOR_COLUMNS = [
    "salary_range",
    "department",
    "company_profile",
    "benefits",
    "requirements",
    "industry",
    "required_education",
    "required_experience"
]


def clean_text(text):
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_data():
  df = pd.read_csv(DATA_PATH)

  for column in MISSING_INDICATOR_COLUMNS:
    df[column + "_missing"] = df[column].isnull().astype(int)

  for column in TEXT_COLUMNS:
    df[column] = df[column].apply(clean_text)

    df[TEXT_COLUMNS] = df[TEXT_COLUMNS].fillna("").astype(str)

    df["combined_text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1)

    df = df.drop(columns=["job_id"], errors="ignore")

    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    Path("dataset/processed").mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data["fraudulent"] = y_train

    test_data = X_test.copy()
    test_data["fraudulent"] = y_test

    train_data.to_csv(
        "dataset/processed/train.csv", index=False
    )

    test_data.to_csv(
        "dataset/processed/test.csv", index=False
    )

    print("\nPREPROCESSING COMPLETED")
    print("-" * 40)

    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))

    print("\nTraining Class Distribution:")
    print(y_train.value_counts())

    print("\nTesting Class Distribution:")
    print(y_test.value_counts())

    print("\nCombined Text Example:")
    print(X_train["combined_text"].iloc[0][:500])

    print("\nProcessed files saved successfully.")


if __name__ == "__main__":
    preprocess_data()
