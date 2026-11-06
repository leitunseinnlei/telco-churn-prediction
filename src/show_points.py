"""
show_points.py
--------------
Prints the starting points and the points for each clue
that logistic regression learned during training.

Run from the project folder (after train_models.py):
    python src/show_points.py
"""

import joblib
import pandas as pd

# Open the trained model, and the training clues (to get the column names)
model = joblib.load("models/logistic_regression.joblib")
X_train = pd.read_csv("data/processed/X_train.csv")

# The starting points (b0)
print(f"Starting points (intercept): {model.intercept_[0]:.3f}\n")

# The points for each clue (b1 to b30), sorted from lowest to highest
points = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
print("Points for each clue (coefficients):")
print(points.round(3).to_string())