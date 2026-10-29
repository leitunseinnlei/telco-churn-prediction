"""
train_models.py
---------------
Step 3 of the pipeline: TRAIN two models on the training data and save them.

Run from the project folder (after data_processing.py):
    python src/train_models.py

Output: 2 trained models saved in the models/ folder.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_training_data():
    """Read the training clues (X_train) and answers (y_train)."""
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    # y_train.csv has one column called "Churn". ["Churn"] takes that column.
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")["Churn"]
    return X_train, y_train


def build_models():
    """Create the two (still untrained) models."""
    models = {
        # Logistic regression: a simple, explainable model.
        # max_iter=1000 gives it enough rounds of practice to finish learning.
        "Logistic regression": LogisticRegression(max_iter=1000),

        # Random forest: many decision trees that vote.
        # n_estimators=300     -> 300 trees.
        # min_samples_leaf=10  -> each final group in a tree must hold at least
        #                         10 customers, so trees can't memorise single people.
        # random_state=42      -> the same result every time we run it.
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, random_state=42
        ),
    }
    return models


def train_models(models, X_train, y_train):
    """Train each model on the training data. This is where the learning happens."""
    for name, model in models.items():
        model.fit(X_train, y_train)  # .fit = "study these customers and their answers"

        # Quick check: how often is the model right on the customers it studied?
        training_accuracy = model.score(X_train, y_train)
        print(f"{name:<20} trained. Accuracy on training data: {training_accuracy:.1%}")
    return models


def save_models(models):
    """Save each trained model as a file, so other scripts can use it later."""
    Path(MODELS_DIR).mkdir(exist_ok=True)  # create the models/ folder if needed
    for name, model in models.items():
        # "Logistic regression" -> "logistic_regression.joblib"
        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, f"{MODELS_DIR}/{file_name}")
        print(f"Saved {MODELS_DIR}/{file_name}")


def run():
    X_train, y_train = load_training_data()
    print(f"Training on {len(X_train)} customers and {X_train.shape[1]} features.\n")

    models = build_models()
    models = train_models(models, X_train, y_train)
    print()
    save_models(models)


if __name__ == "__main__":
    run()