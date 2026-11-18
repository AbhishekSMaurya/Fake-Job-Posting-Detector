
import pandas as pd

df = pd.read_csv("dataset/fake_job_postings.csv")

features = [
    "salary_range",
    "department",
    "company_profile",
    "benefits",
    "requirements",
    "industry",
    "required_education",
    "required_experience"
]

for feature in features:
    df[feature + "_missing"] = df[feature].isnull().astype(int)

print("\nMISSING INFORMATION VS FRAUD")
print("-" * 60)

for feature in features:
    print(f"\nFeature: {feature}")

    result = df.groupby(feature + "_missing")["fraudulent"].agg(
        ["count", "mean"]
    )

    result["fraud_rate_percentage"] = result["mean"] * 100

    result.index = ["Present", "Missing"]

    print(result)
