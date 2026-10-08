"""
Preprocessing & Feature Engineering module for Employee Retention Prediction System.
Follows strict featurization ordering (Split BEFORE Fit) to prevent data leakage.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hr_employee_attrition.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Columns to drop (zero variance / metadata identifiers)
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived organizational & behavioral features without leaking target."""
    df = df.copy()
    
    # 1. Career stability & loyalty index
    df["TenurePerJob"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    
    # 2. Manager alignment ratio
    df["YearsWithManagerRatio"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    
    # 3. Role stagnation ratio
    df["YearsInRoleRatio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    
    # 4. Income efficiency per experience year
    df["IncomePerWorkingYear"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    
    # 5. Composite satisfaction score (average of the 4 surveys)
    df["CompositeSatisfaction"] = (
        df["EnvironmentSatisfaction"] +
        df["JobSatisfaction"] +
        df["RelationshipSatisfaction"] +
        df["WorkLifeBalance"]
    ) / 4.0
    
    # 6. Promotion stagnation indicator
    df["YearsWithoutPromotionRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    
    return df

def get_preprocessor(categorical_features, numerical_features):
    """Constructs a scikit-learn ColumnTransformer."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    return preprocessor

def prepare_data(test_size=0.20, random_state=42):
    """
    Loads dataset, drops zero-variance columns, creates engineered features,
    and performs a stratified split into train and test sets BEFORE fitting.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # Drop zero variance
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Target variable
    y = (df["Attrition"] == "Yes").astype(int)
    X_raw = df.drop(columns=["Attrition"])
    
    # Feature engineering applied sample-by-sample
    X_feat = engineer_features(X_raw)
    
    # STRICT FEATURIZATION ORDERING: Split BEFORE fit
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    categorical_features = [
        col for col in X_feat.select_dtypes(include=["object"]).columns
    ]
    numerical_features = [
        col for col in X_feat.select_dtypes(include=[np.number]).columns
    ]
    
    preprocessor = get_preprocessor(categorical_features, numerical_features)
    
    # FIT solely on X_train_raw
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_cat_names
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test_raw.index)
    
    # Persist the preprocessor and feature schema
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump({
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "feature_names": feature_names,
        "drop_cols": cols_to_drop
    }, os.path.join(MODELS_DIR, "feature_metadata.joblib"))
    
    print(f"Data prepared successfully:")
    print(f"Training set: {X_train_df.shape}, Positive rate: {y_train.mean():.2%}")
    print(f"Test set:     {X_test_df.shape}, Positive rate: {y_test.mean():.2%}")
    print(f"Total features after one-hot encoding: {len(feature_names)}")
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    prepare_data()
