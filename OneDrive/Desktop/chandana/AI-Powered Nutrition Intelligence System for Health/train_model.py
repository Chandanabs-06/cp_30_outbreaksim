
import pandas as pd
import numpy as np
import joblib
import traceback

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. PROJECT PATH AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# Use the original dataset to avoid using encodings
# fitted on the entire dataset before the split.
DATASET_PATH = BASE_DIR / "nutrition_symptom_patient_dataset.csv"

TARGET = "primary_deficiency"


def main():

    print("=" * 60)
    print("NUTRITIONAL DEFICIENCY PREDICTION")
    print("=" * 60)

    # ==================================================
    # 2. LOAD DATASET
    # ==================================================

    print("\n[1/7] Loading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n"
            "Place the CSV file in the same folder as this script."
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove rows without a target label
    df = df.dropna(subset=[TARGET]).copy()

    if df.empty:
        raise ValueError("No rows with a valid target label remain.")

    # ==================================================
    # 3. SEPARATE FEATURES AND TARGET
    # ==================================================

    print("\n[2/7] Separating features and target...")

    y = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET]).copy()

    # Exclude identifier columns
    identifier_columns = [
        col for col in X.columns
        if col.strip().lower() in [
            "patient_id",
            "user_id",
            "record_id",
            "id"
        ]
    ]

    X = X.drop(columns=identifier_columns, errors="ignore")

    # Remove columns containing only missing values
    empty_columns = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty_columns, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    # Remove rows with missing target values has already happened.
    # Drop constant columns because they add no predictive information.
    constant_columns = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=constant_columns, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("All feature columns are empty or constant.")

    print("Excluded identifier columns:", identifier_columns)
    print("Excluded empty columns:", empty_columns)
    print("Excluded constant columns:", constant_columns)
    print("Number of features:", X.shape[1])
    print("Number of classes:", y.nunique())

    print("\nTarget class counts:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two classes to train a classifier."
        )

    # ==================================================
    # 4. TRAIN-TEST SPLIT
    # ==================================================

    print("\n[3/7] Splitting dataset into training and testing sets...")

    # Stratification requires enough examples of each class.
    class_counts = y.value_counts()
    test_rows = int(np.ceil(len(y) * 0.20))
    train_rows = len(y) - test_rows

    can_stratify = (
        class_counts.min() >= 2
        and test_rows >= y.nunique()
        and train_rows >= y.nunique()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if can_stratify else None
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # ==================================================
    # 5. PREPROCESS FEATURES
    # ==================================================

    print("\n[4/7] Preparing preprocessing pipeline...")

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Ensure any unusual column types are also handled
    other_columns = [
        col for col in X_train.columns
        if col not in numerical_columns
        and col not in categorical_columns
    ]

    categorical_columns.extend(other_columns)

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numeric", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # ==================================================
    # 6. TRAIN MODEL
    # ==================================================

    print("\n[5/7] Training Random Forest model...")
    print("Please wait...")

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    print("Training completed successfully!")

    # ==================================================
    # 7. PREDICT AND EVALUATE
    # ==================================================

    print("\n[6/7] Evaluating model on test data...")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test accuracy percentage: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(
        f"Balanced accuracy percentage: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=pipeline.classes_))

    # Save actual and predicted values
    results = pd.DataFrame({
        "actual_deficiency": y_test.to_numpy(),
        "predicted_deficiency": y_pred
    })

    results.to_csv(
        BASE_DIR / "model_test_predictions.csv",
        index=False
    )

    # ==================================================
    # 8. SAVE MODEL AND RESULTS
    # ==================================================

    print("\n[7/7] Saving trained model...")

    model_path = BASE_DIR / "nutrition_deficiency_model.joblib"
    joblib.dump(pipeline, model_path)

    metrics = pd.DataFrame([{
        "model": "Random Forest",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "number_of_features": X.shape[1],
        "number_of_classes": y.nunique(),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "balanced_accuracy": balanced_accuracy
    }])

    metrics.to_csv(
        BASE_DIR / "model_evaluation_metrics.csv",
        index=False
    )

    print("\nSaved model:", model_path.name)
    print("Saved predictions: model_test_predictions.csv")
    print("Saved metrics: model_evaluation_metrics.csv")

    print("\n" + "=" * 60)
    print("TRAINING AND TESTING COMPLETED!")
    print(f"FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nMODEL EXECUTION FAILED")
        print("Error:", error)
        traceback.print_exc()
