
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# 1. SET DATASET PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PATIENT_FILE = BASE_DIR / "nutrition_symptom_patient_dataset.csv"
FOOD_FILE = BASE_DIR / "food_diary_nutrient_intake_dataset.csv"


# --------------------------------------------------
# 2. FUNCTION TO PREPROCESS A DATASET
# --------------------------------------------------

def preprocess_dataset(file_path, output_file, mapping_file,
                       age_grouping=False):

    print("\n" + "=" * 60)
    print(f"Processing: {file_path.name}")
    print("=" * 60)

    # Check whether the dataset exists
    if not file_path.exists():
        print(f"ERROR: Dataset not found: {file_path}")
        print("Place the CSV file in the same folder as this script.")
        return None

    # Load dataset
    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    # --------------------------------------------------
    # 3. CLEAN COLUMN NAMES
    # --------------------------------------------------

    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    print("\nDuplicate rows removed:", duplicate_count)

    # Replace infinite numerical values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --------------------------------------------------
    # 4. HANDLE AGE VALUES AND CREATE AGE GROUPS
    # --------------------------------------------------

    if age_grouping and "age" in df.columns.str.lower().tolist():

        # Find the actual age column, ignoring capitalization
        age_column = next(
            col for col in df.columns
            if col.lower() == "age"
        )

        df[age_column] = pd.to_numeric(
            df[age_column], errors="coerce"
        )

        # Fill missing age values using the median
        if df[age_column].notna().any():
            df[age_column] = df[age_column].fillna(
                df[age_column].median()
            )

        # Group ages
        df["age_group"] = pd.cut(
            df[age_column],
            bins=[-1, 12, 19, 59, np.inf],
            labels=[
                "Child",
                "Teenager",
                "Adult",
                "Older Adult"
            ],
            right=True
        )

        print("\nAge groups created:")
        print(df["age_group"].value_counts(dropna=False))

    # --------------------------------------------------
    # 5. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
    # --------------------------------------------------

    # Convert boolean columns into text categories
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].map({
                True: "True",
                False: "False"
            })

    categorical_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # --------------------------------------------------
    # 6. HANDLE MISSING VALUES
    # --------------------------------------------------

    # Fill numerical missing values with the median
    for col in numerical_cols:
        if df[col].isna().any() and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with the most frequent value
    for col in categorical_cols:
        if df[col].isna().any() and df[col].notna().any():
            mode_values = df[col].mode(dropna=True)

            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])

    # Remove columns that contain only missing values
    all_missing_cols = df.columns[df.isna().all()].tolist()

    if all_missing_cols:
        print("\nRemoving entirely empty columns:", all_missing_cols)
        df.drop(columns=all_missing_cols, inplace=True)

    # Refresh categorical columns after missing-value handling
    categorical_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # --------------------------------------------------
    # 7. LABEL ENCODING
    # --------------------------------------------------

    encoding_mappings = []

    for col in categorical_cols:

        # Convert to object to prevent pandas string-dtype errors
        df[col] = df[col].astype(object)

        valid_mask = df[col].notna()

        if not valid_mask.any():
            continue

        # Convert values to strings before encoding
        values = df.loc[valid_mask, col].astype(str)

        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(values)

        # Assign encoded integers safely
        df.loc[valid_mask, col] = encoded_values

        # Save the original-to-encoded mapping
        for original_value, encoded_value in zip(
            encoder.classes_,
            range(len(encoder.classes_))
        ):
            encoding_mappings.append({
                "column": col,
                "original_value": original_value,
                "encoded_value": encoded_value
            })

        # Convert encoded column to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save encoding mappings
    pd.DataFrame(
        encoding_mappings,
        columns=["column", "original_value", "encoded_value"]
    ).to_csv(mapping_file, index=False)

    # --------------------------------------------------
    # 8. SAVE PREPROCESSED DATASET
    # --------------------------------------------------

    df.to_csv(output_file, index=False)

    print("\nPreprocessing completed successfully!")
    print("Final dataset shape:", df.shape)
    print("Saved processed dataset:", output_file.name)
    print("Saved encoding mappings:", mapping_file.name)

    print("\nProcessed dataset preview:")
    print(df.head())

    print("\nRemaining missing values:", int(df.isna().sum().sum()))

    return df


# --------------------------------------------------
# 9. PREPROCESS BOTH DATASETS
# --------------------------------------------------

if __name__ == "__main__":

    # Patient symptoms and nutritional deficiencies
    patient_df = preprocess_dataset(
        file_path=PATIENT_FILE,
        output_file=BASE_DIR / "preprocessed_patient_dataset.csv",
        mapping_file=BASE_DIR / "patient_encoding_mappings.csv",
        age_grouping=True
    )

    # Food diary and nutrient intake
    food_df = preprocess_dataset(
        file_path=FOOD_FILE,
        output_file=BASE_DIR / "preprocessed_food_diary_dataset.csv",
        mapping_file=BASE_DIR / "food_encoding_mappings.csv",
        age_grouping=False
    )

    print("\n" + "=" * 60)

    if patient_df is not None and food_df is not None:
        print("ALL DATASETS PREPROCESSED SUCCESSFULLY!")
    else:
        print("Preprocessing finished with an error.")
        print("Check the file names and folder paths.")

    print("=" * 60)
