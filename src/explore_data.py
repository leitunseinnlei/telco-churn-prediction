"""
explore_data.py
---------------
Step 1 of the pipeline: LOOK at the data before changing anything.

Run from the project folder:
    python src/explore_data.py

Output: a summary in the terminal + 3 charts in the results/ folder.
"""

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
# read_csv opens the CSV file and turns it into a table.
# In pandas, a table is called a "DataFrame". We store it in a variable called df.
df = pd.read_csv("data/raw/telco_churn.csv")

# ------------------------------------------------------------
# 2. How big is the table?
# ------------------------------------------------------------
# df.shape gives (number of rows, number of columns).
rows, columns = df.shape
print(f"The table has {rows} rows (customers) and {columns} columns.\n")

# ------------------------------------------------------------
# 3. What do the first rows look like?
# ------------------------------------------------------------
# df.head() shows the first 5 rows. We pick a few columns so it fits on screen.
print("First 5 customers (a few columns):")
print(df[["customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"]].head())
print()

# ------------------------------------------------------------
# 4. What type of data is in each column?
# ------------------------------------------------------------
# Numbers can be used in calculations. Text cannot.
print("Column types:")
print(df.dtypes)
print()

# ------------------------------------------------------------
# 5. Are there missing values?
# ------------------------------------------------------------
# isna() marks empty cells, and sum() counts them per column.
missing = df.isna().sum()
print(f"Empty cells found by pandas: {missing.sum()}")

# Careful: some files store "missing" as a blank space " " instead of an empty cell.
# pandas does not see a space as missing, so we check for it ourselves.
blank_spaces = (df["TotalCharges"].str.strip() == "").sum()
print(f"Blank spaces hiding in TotalCharges: {blank_spaces}\n")

# ------------------------------------------------------------
# 6. What are we predicting? (the target column: Churn)
# ------------------------------------------------------------
# value_counts counts how many times each answer appears.
print("How many customers left (Churn = Yes)?")
print(df["Churn"].value_counts())
print()
churn_rate = (df["Churn"] == "Yes").mean()
print(f"Churn rate: {churn_rate:.1%} of customers left.\n")

# ------------------------------------------------------------
# 7. Charts: which customers leave more often?
# ------------------------------------------------------------
# Helper column: 1 if the customer left, 0 if they stayed.
# The average of a 0/1 column = the share of 1s = the churn rate.
df["left"] = (df["Churn"] == "Yes").astype(int)

# --- Chart 1: churn rate by contract type ---
contract_rate = df.groupby("Contract")["left"].mean() * 100
print("Churn rate by contract type (%):")
print(contract_rate.round(1))
print()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(contract_rate.index, contract_rate.values, color="#378ADD")
ax.set_ylabel("Churn rate (%)")
ax.set_title("Month-to-month customers leave most often")
plt.tight_layout()
plt.savefig("results/eda_churn_by_contract.png", dpi=150)
plt.close()

# --- Chart 2: churn rate by how long someone has been a customer ---
# pd.cut puts each customer's tenure (months) into a group.
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24", "25-48", "49-72"],
)
tenure_rate = df.groupby("tenure_group", observed=True)["left"].mean() * 100
print("Churn rate by tenure group (%):")
print(tenure_rate.round(1))
print()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(tenure_rate.index.astype(str), tenure_rate.values, color="#378ADD")
ax.set_ylabel("Churn rate (%)")
ax.set_xlabel("Months as a customer")
ax.set_title("New customers leave far more often than loyal ones")
plt.tight_layout()
plt.savefig("results/eda_churn_by_tenure.png", dpi=150)
plt.close()

# --- Chart 3: monthly bill for customers who stayed vs left ---
stayed = df.loc[df["Churn"] == "No", "MonthlyCharges"]
left = df.loc[df["Churn"] == "Yes", "MonthlyCharges"]
print(f"Median monthly bill - stayed: {stayed.median():.2f}, left: {left.median():.2f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(stayed, bins=30, alpha=0.6, label="Stayed", color="#888780")
ax.hist(left, bins=30, alpha=0.6, label="Left", color="#D85A30")
ax.set_xlabel("Monthly charges")
ax.set_ylabel("Number of customers")
ax.set_title("Customers who left often had higher monthly bills")
ax.legend()
plt.tight_layout()
plt.savefig("results/eda_monthly_charges.png", dpi=150)
plt.close()

print("\nSaved 3 charts in the results/ folder.")