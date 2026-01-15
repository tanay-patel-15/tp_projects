"""
Data preprocessing module for diabetes prediction.
Handles missing values, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load diabetes dataset from CSV file."""
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    Columns like Glucose, BloodPressure, SkinThickness, Insulin, BMI
    cannot be zero - treat zeros as missing and impute with median.
    """
    df = df.copy()
    
    # Columns where 0 is not a valid value
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_not_valid:
        if col in df.columns:
            # Replace zeros with NaN
            df[col] = df[col].replace(0, np.nan)
            # Impute with median
            df[col] = df[col].fillna(df[col].median())
    
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale features using StandardScaler.
    Fits on training data and transforms both train and test.
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(df: pd.DataFrame, target_col: str = 'Outcome', 
                 test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Complete data preparation pipeline.
    
    Args:
        df: Raw dataframe
        target_col: Name of target column
        test_size: Proportion for test split
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train/test splits and scaler
    """
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Split features and target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
