"""
evaluate.py
-----------
Step 4 of the pipeline: TEST both trained models on the hidden test customers,
compare them with several metrics, and save charts and a comparison table.

Run from the project folder (after train_models.py):
    python src/evaluate.py

Output: results/model_comparison.csv + 3 charts in results/.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

MODEL_FILES = {
    "Logistic regression": "logistic_regression.joblib",
    "Random forest": "random_forest.joblib",
}


def load_test_data():
    """Read the hidden test customers: their clues (X) and true answers (y)."""
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")["Churn"]
    return X_test, y_test


def load_models():
    """Load the trained models that train_models.py saved."""
    return {name: joblib.load(f"{MODELS_DIR}/{file}") for name, file in MODEL_FILES.items()}


def calculate_metrics(y_true, y_pred, y_chance):
    """Calculate the five metrics for one model."""
    return {
        # Of all customers, how many did we get right?
        "Accuracy": accuracy_score(y_true, y_pred),
        # Of the customers we said would leave, how many really left?
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        # Of the customers who really left, how many did we find?
        "Recall": recall_score(y_true, y_pred),
        # One number that balances precision and recall.
        "F1 score": f1_score(y_true, y_pred),
        # How well the model RANKS leavers above stayers (0.5 = guessing, 1 = perfect).
        "ROC AUC": roc_auc_score(y_true, y_chance),
    }


def evaluate_models(models, X_test, y_test):
    """Score every model (plus a lazy baseline) and return a comparison table."""
    results = {}

    # Baseline: a "lazy model" that always says "will stay" (0).
    # Any real model must beat this to be useful.
    lazy_pred = [0] * len(y_test)
    lazy_chance = [0.0] * len(y_test)
    results["Baseline (always 'stay')"] = calculate_metrics(y_test, lazy_pred, lazy_chance)

    for name, model in models.items():
        y_pred = model.predict(X_test)                  # yes/no answer (threshold 50%)
        y_chance = model.predict_proba(X_test)[:, 1]    # chance of leaving, 0 to 1
        results[name] = calculate_metrics(y_test, y_pred, y_chance)

    # Turn the dictionary into a table: one row per model, one column per metric.
    return pd.DataFrame(results).T.round(3)


def plot_confusion_matrices(models, X_test, y_test):
    """Chart 1: one confusion matrix per model, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["Stayed", "Left"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("What the model said")
        ax.set_ylabel("What really happened")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    """Chart 2: ROC curves of both models on one chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
    ax.set_title("ROC curves: closer to the top-left corner is better")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=150)
    plt.close()


def plot_top_features(models, feature_names):
    """Chart 3: the 10 most important features of each model."""
    lr = models["Logistic regression"]
    rf = models["Random forest"]

    # Logistic regression: the size of each coefficient (ignoring + or -).
    lr_importance = pd.Series(abs(lr.coef_[0]), index=feature_names).nlargest(10)
    # Random forest: total Gini decrease per feature.
    rf_importance = pd.Series(rf.feature_importances_, index=feature_names).nlargest(10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(lr_importance.index[::-1], lr_importance.values[::-1], color="#378ADD")
    axes[0].set_title("Logistic regression: size of coefficient")
    axes[1].barh(rf_importance.index[::-1], rf_importance.values[::-1], color="#1D9E75")
    axes[1].set_title("Random forest: feature importance")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/top_features.png", dpi=150)
    plt.close()


def run():
    X_test, y_test = load_test_data()
    models = load_models()
    print(f"Testing on {len(X_test)} hidden customers ({y_test.sum()} of them left).\n")

    comparison = evaluate_models(models, X_test, y_test)
    print("Model comparison on the TEST data:")
    print(comparison.to_string())
    comparison.to_csv(f"{RESULTS_DIR}/model_comparison.csv")

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_top_features(models, X_test.columns)

    print(f"\nSaved model_comparison.csv and 3 charts in {RESULTS_DIR}/")


if __name__ == "__main__":
    run()