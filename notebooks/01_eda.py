# Exploratory Data Analysis — Credit Risk Dataset
#
# This is a Python-script form of the EDA notebook. Convert to .ipynb with:
#     jupyter nbconvert --to notebook --execute 01_eda.py
#
# Or import sections interactively.

# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Load data
df = pd.read_csv("../data/raw/lending_club.csv")
print(f"Shape: {df.shape}")
df.head()

# %% Target distribution
ax = df["loan_status"].value_counts(normalize=True).plot(kind="bar")
ax.set_title("Class balance")
plt.show()

# %% Missing values
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].head(20))

# %% Numerical correlations
num_cols = df.select_dtypes(include="number").columns
corr = df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False)
plt.title("Numerical feature correlations")
plt.show()

# %% Default rate by loan grade
grade_default = df.groupby("grade")["loan_status"].apply(
    lambda x: (x == "Charged Off").mean()
)
grade_default.plot(kind="bar", title="Default rate by grade")
plt.show()

# %% Default rate vs interest rate
df["int_rate_num"] = df["int_rate"].astype(str).str.rstrip("%").astype(float)
df["int_rate_bin"] = pd.cut(df["int_rate_num"], bins=10)
df.groupby("int_rate_bin")["loan_status"].apply(
    lambda x: (x == "Charged Off").mean()
).plot(kind="bar", title="Default rate vs interest rate bin")
plt.show()

# %% DTI distribution split by status
plt.figure(figsize=(10, 5))
for status, sub in df.groupby("loan_status"):
    sns.kdeplot(sub["dti"], label=status, fill=True, alpha=0.3)
plt.legend()
plt.title("DTI distribution by loan status")
plt.show()
