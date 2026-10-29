"""
data_processing.py
------------------
Step 2 of the pipeline: CLEAN the data, turn it into numbers,
SPLIT it into training and test sets, and SCALE the number columns.

Run from the project folder:
    python src/data_processing.py

Output: 4 files in data/processed/ (X_train, X_test, y_train, y_test).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_DIR = "data/processed"

# The columns with real amounts (not 0/1 flags). Only these need scaling.
NUMBER_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ------------------------------------------------------------
# A "function" is a named recipe. We write it once with "def",
# and later we can use it again by calling its name.
# Each function below does ONE job.
# ------------------------------------------------------------


def load_data(path):
    """Read the CSV file and return it as a table (DataFrame)."""
    return pd.read_csv(path)


def clean_data(df):
    """Fix missing values, remove useless columns, and turn the target into 0/1."""
    # Work on a copy, so the original table stays untouched.
    df = df.copy()

    # --- 1. Fix the hidden missing values in TotalCharges ---
    # pd.to_numeric turns text like "1889.5" into the number 1889.5.
    # errors="coerce" means: if a value can't become a number (like " "),
    # mark it as missing (NaN) instead of crashing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"Missing TotalCharges after converting to numbers: {df['TotalCharges'].isna().sum()}")

    # These are brand-new customers (tenure = 0) who have not been billed yet,
    # so their real total is 0. fillna replaces missing values.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Missing TotalCharges after filling with 0: {df['TotalCharges'].isna().sum()}")

    # --- 2. Remove the customer ID ---
    # An ID is just a name tag. It says nothing about whether someone will leave,
    # and a model could wrongly try to learn from it.
    df = df.drop(columns=["customerID"])

    # --- 3. Turn the target into numbers: Yes -> 1, No -> 0 ---
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_features(df):
    """Turn text columns into number columns (one-hot encoding)."""
    # Find all columns that still contain text.
    text_columns = df.select_dtypes(include="str").columns.tolist()
    print(f"Text columns to encode: {len(text_columns)}")

    # get_dummies creates one new 0/1 column per category.
    # Example: Contract -> Contract_One year, Contract_Two year.
    # drop_first=True removes one category per column, because it is not needed:
    # if both new columns are 0, the contract must be "Month-to-month".
    df = pd.get_dummies(df, columns=text_columns, drop_first=True, dtype=int)
    return df


def split_data(df):
    """Split into features (X) and target (y), then into training and test sets."""
    # X = the clues the model learns from. y = the answer it must predict.
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # test_size=0.2      -> 20% of customers are kept aside as the "final exam".
    # stratify=y         -> both parts keep the same churn rate (about 26.5%).
    # random_state=42    -> the same random split every time we run the script.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Put the number columns on the same scale (average 0, spread 1)."""
    # Work on copies, so we don't change the tables outside this function.
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    # fit_transform on the TRAINING data: learn the average and spread, then scale.
    X_train[NUMBER_COLUMNS] = scaler.fit_transform(X_train[NUMBER_COLUMNS])

    # transform on the TEST data: use the SAME average and spread from training.
    # We never "fit" on the test data. That would let the model peek at the exam.
    X_test[NUMBER_COLUMNS] = scaler.transform(X_test[NUMBER_COLUMNS])

    return X_train, X_test


def save_data(X_train, X_test, y_train, y_test):
    """Save the four ready-to-use tables in data/processed/."""
    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)


def run():
    """Run all the processing steps in order and show what changed."""
    df = load_data(RAW_PATH)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns\n")

    df = clean_data(df)
    print(f"\nAfter cleaning: {df.shape[0]} rows, {df.shape[1]} columns\n")

    df = encode_features(df)
    print(f"After encoding: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # --- Split ---
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set: {len(X_train)} customers, churn rate {y_train.mean():.1%}")
    print(f"Test set:     {len(X_test)} customers, churn rate {y_test.mean():.1%}\n")

    # --- Scale ---
    print("Number columns BEFORE scaling (training set):")
    print(X_train[NUMBER_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(2))
    print()

    X_train, X_test = scale_features(X_train, X_test)

    print("Number columns AFTER scaling (training set):")
    print(X_train[NUMBER_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(2))
    print()

    # --- Save ---
    save_data(X_train, X_test, y_train, y_test)
    print(f"Saved X_train, X_test, y_train, y_test in {PROCESSED_DIR}/")


# This line means: only call run() when this file is started directly
# (python src/data_processing.py), not when another script borrows its functions.
if __name__ == "__main__":
    run()