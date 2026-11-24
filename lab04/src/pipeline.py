import hashlib
import os
import random
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import yaml

# ==========================================
# 0. Global Seeds & Version Logging
# ==========================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Python Version: {sys.version}")
print(f"Pandas Version: {pd.__version__}")


# ==========================================
# 1. Ingestion Step
# ==========================================
def ingest_data():
    print("Ingesting Adult dataset from OpenML...")
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    
    # Log row and column counts as required
    print(f"Raw Ingestion Log -> Row Count: {df.shape[0]}, Column Count: {df.shape[1]}")
    assert df.shape[0] == 48842, f"Expected 48842 rows, got {df.shape[0]}"
    assert df.shape[1] == 15, f"Expected 15 columns, got {df.shape[1]}"
    return df


# ==========================================
# 2. Validation Step
# ==========================================
def validate_schema(df):
    print("Running schema validation...")
    violations = 0
    
    # Check null policy ('?' represents missing values)
    # Report violations without silently dropping rows
    for col in df.columns:
        if df[col].dtype == object:
            missing_count = (df[col] == '?').sum() + df[col].isna().sum()
            if missing_count > 0:
                print(f"[Validation Notice] Column '{col}' contains {missing_count} missing/ '?' entries.")
        
    print(f"Validation completed. Total violation notices reported: {violations}")


# ==========================================
# 3 & 4. Cleaning, Splitting & Transformation
# ==========================================
def run_pipeline(df):
    # Standardize target column name if necessary
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    
    # Convert target to binary (1 for '>50K', 0 otherwise)
    y = df[target_col].apply(lambda val: 1 if '>50K' in str(val) else 0)
    X = df.drop(columns=[target_col])

    # Split train/test 70/30 stratified BEFORE cleaning to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    # Clean missing values ('?' to NaN)
    X_train = X_train.replace('?', np.nan)
    X_test = X_test.replace('?', np.nan)

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['category', object]).columns.tolist()

    # Fit imputer on train only (Median for numeric, Most Frequent/Mode for categorical)
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # Outliers: clip numeric features to 1st/99th percentile from train
    lower_bounds = X_train[numeric_cols].quantile(0.01)
    upper_bounds = X_train[numeric_cols].quantile(0.99)
    
    X_train[numeric_cols] = X_train[numeric_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    X_test[numeric_cols] = X_test[numeric_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    # Drop exact duplicates from train only (keep test untouched)
    train_df = pd.concat([X_train, y_train], axis=1)
    initial_train_len = len(train_df)
    train_df = train_df.drop_duplicates()
    print(f"Dropped {initial_train_len - len(train_df)} exact duplicate rows from training set.")
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # ==========================================
    # 5. Feature Engineering
    # ==========================================
    def engineer_features(frame):
        f = frame.copy()
        # capital_net = capital-gain - capital-loss
        gain_col = 'capital-gain' if 'capital-gain' in f.columns else 'capital_gain'
        loss_col = 'capital-loss' if 'capital-loss' in f.columns else 'capital_loss'
        f['capital_net'] = f[gain_col] - f[loss_col]
        
        # hours_per_age = hours-per-week / (age - 16)
        hours_col = 'hours-per-week' if 'hours-per-week' in f.columns else 'hours_per_week'
        f['hours_per_age'] = f[hours_col] / (f['age'] - 16).replace(0, 0.1) # prevent division by zero
        return f

    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)

    # Save fitted transformers/preprocessor state
    os.makedirs("lab04/results", exist_ok=True)
    joblib.dump({
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }, "lab04/results/preprocessor.joblib")
    print("Saved preprocessor to lab04/results/preprocessor.joblib")

    return X_train, X_test, y_train, y_test, X_train_eng, X_test_eng


# ==========================================
# 6. Label-Quality Audit Check
# ==========================================
def audit_label_quality(X_raw, y_raw):
    print("\n--- Running Label-Quality Conflict Audit ---")
    combined = X_raw.copy()
    combined['target'] = y_raw.values
    
    # Find duplicate feature rows with conflicting labels
    feature_cols = [c for c in combined.columns if c != 'target']
    duplicates = combined.duplicated(subset=feature_cols, keep=False)
    conflicting_rows = combined[duplicates]
    
    conflict_groups = conflicting_rows.groupby(feature_cols)['target'].nunique()
    true_conflicts = conflict_groups[conflict_groups > 1]
    
    print(f"Total rows involved in duplicate feature groups: {len(conflicting_rows)}")
    print(f"Number of distinct feature sets with conflicting labels: {len(true_conflicts)}")


# ==========================================
# 7. Hashing & Metadata Configuration
# ==========================================
def compute_sha256_and_config(df):
    data_bytes = df.to_csv(index=False).encode('utf-8')
    dataset_hash = hashlib.sha256(data_bytes).hexdigest()
    print(f"Processed Dataset SHA256 Hash: {dataset_hash}")

    config_data = {
        "dataset": "Adult Census Income (OpenML)",
        "sha256_hash": dataset_hash,
        "random_seed": RANDOM_SEED,
        "test_size": 0.30
    }
    
    with open("lab04/results/config.yaml", "w") as f:
        yaml.dump(config_data, f)
    print("Saved configuration YAML to lab04/results/config.yaml")


if __name__ == "__main__":
    df = ingest_data()
    validate_schema(df)
    
    X_train, X_test, y_train, y_test, X_train_eng, X_test_eng = run_pipeline(df)
    
    audit_label_quality(X_train, y_train)
    compute_sha256_and_config(df)