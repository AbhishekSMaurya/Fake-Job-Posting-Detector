
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/fake_job_postings.csv")

sns.set_theme(style="whitegrid")

print("DATASET OVERVIEW")
print("-" * 40)
print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nCLASS DISTRIBUTION")
print("-" * 40)
print(df["fraudulent"].value_counts())

print("\nCLASS DISTRIBUTION PERCENTAGE")
print("-" * 40)
print(df["fraudulent"].value_counts(normalize=True) * 100)

print("\nMISSING VALUE PERCENTAGE")
print("-" * 40)
missing = (df.isnull().sum() / len(df)) * 100
print(missing[missing > 0].sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x="fraudulent", ax=axes[0])
axes[0].set_title("Genuine vs Fraudulent Jobs")
axes[0].set_xticks([0, 1], ["Genuine", "Fraudulent"])
axes[0].set_xlabel("Job Type")
axes[0].set_ylabel("Number of Postings")

missing.sort_values(ascending=False).head(12).plot(
    kind="bar", ax=axes[1]
)
axes[1].set_title("Top 12 Features with Missing Values")
axes[1].set_ylabel("Missing Percentage (%)")
axes[1].set_xlabel("Features")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("dataset_overview.png", dpi=300)
plt.close()

print("\nFRAUD RATE BY COMPANY LOGO")
print("-" * 40)

logo_analysis = df.groupby("has_company_logo")["fraudulent"].agg(
    ["count", "mean"]
)
logo_analysis["fraud_rate_percentage"] = logo_analysis["mean"] * 100
print(logo_analysis)

print("\nFRAUD RATE BY TELECOMMUTING")
print("-" * 40)

telecommuting_analysis = df.groupby("telecommuting")["fraudulent"].agg(
    ["count", "mean"]
)
telecommuting_analysis["fraud_rate_percentage"] = (
    telecommuting_analysis["mean"] * 100
)
print(telecommuting_analysis)

print("\nTOP INDUSTRIES BY FRAUD RATE")
print("-" * 40)

industry_analysis = df.groupby("industry")["fraudulent"].agg(
    ["count", "mean"]
)

industry_analysis = industry_analysis[
    industry_analysis["count"] >= 100
]

industry_analysis["fraud_rate_percentage"] = (
    industry_analysis["mean"] * 100
)

print(
    industry_analysis.sort_values(
        "fraud_rate_percentage", ascending=False
    ).head(10)
)

print("\nEDA COMPLETED SUCCESSFULLY")
