import pandas as pd

df = pd.read_csv("dataset/fake_job_postings.csv")
df=df.drop_duplicates()
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["fraudulent"].value_counts())