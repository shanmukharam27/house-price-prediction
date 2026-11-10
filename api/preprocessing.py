import pandas as pd
from src.preprocess_pipeline import engineer_features

def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Ames Housing data and apply domain feature engineering before prediction.
    """
    df = df.copy()

    # Remove ID because it is not a feature
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Remove target column if accidentally provided
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])

    # Apply feature engineering
    return engineer_features(df)
