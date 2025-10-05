import pandas as pd
from src.config import FEATURE_COLUMNS, TARGET_COLUMN

def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extrai as features (X) e o target (y) do DataFrame.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
