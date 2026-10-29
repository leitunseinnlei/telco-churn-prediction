"""
main.py
-------
Runs the WHOLE pipeline with one command:
    1. data_processing.py -> clean, encode, split, scale, save
    2. train_models.py    -> train and save both models
    3. evaluate.py        -> test both models, save the table and charts

Run from the project folder:
    python src/main.py

(explore_data.py is not part of the pipeline: it is a one-off look at the data.
Run it on its own with: python src/explore_data.py)
"""

# Each script has a function called run(). We borrow them here.
# "as" gives each one a clearer name, because they are all called "run".
from data_processing import run as process_data
from evaluate import run as evaluate_models
from train_models import run as train_models


def main():
    print("=" * 60)
    print("STEP 1 of 3: Data processing")
    print("=" * 60)
    process_data()

    print("\n" + "=" * 60)
    print("STEP 2 of 3: Training the models")
    print("=" * 60)
    train_models()

    print("\n" + "=" * 60)
    print("STEP 3 of 3: Evaluating the models")
    print("=" * 60)
    evaluate_models()

    print("\nPipeline finished. Open the results/ folder to see the outputs.")


if __name__ == "__main__":
    main()