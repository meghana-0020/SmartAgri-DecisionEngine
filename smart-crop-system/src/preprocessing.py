from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    X = df[
        [
            "Soil_Type",
            "Rainfall",
            "Temperature",
            "Humidity",
            "Fertilizer_Usage",
            "Pesticide_Usage",
            "Irrigation_Hours",
            "Previous_Yield",
            "Season",
        ]
    ]

    targets = {
        "yield": df["Crop_Yield"],
        "health": df["Crop_Health"],
        "suitability": df["Suitability"],
        "productivity": df["Productivity_Zone"],
    }
    return X, targets


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_features = ["Soil_Type", "Season"]
    numeric_features = [
        "Rainfall",
        "Temperature",
        "Humidity",
        "Fertilizer_Usage",
        "Pesticide_Usage",
        "Irrigation_Hours",
        "Previous_Yield",
    ]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor


def train_test_split_all(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, targets = get_feature_targets(df)

    X_train, X_test, y_yield_train, y_yield_test = train_test_split(
        X,
        targets["yield"],
        test_size=test_size,
        random_state=random_state,
    )

    _, _, y_health_train, y_health_test = train_test_split(
        X,
        targets["health"],
        test_size=test_size,
        random_state=random_state,
    )

    _, _, y_suit_train, y_suit_test = train_test_split(
        X,
        targets["suitability"],
        test_size=test_size,
        random_state=random_state,
    )

    _, _, y_prod_train, y_prod_test = train_test_split(
        X,
        targets["productivity"],
        test_size=test_size,
        random_state=random_state,
    )

    return (
        X_train,
        X_test,
        y_yield_train,
        y_yield_test,
        y_health_train,
        y_health_test,
        y_suit_train,
        y_suit_test,
        y_prod_train,
        y_prod_test,
    )
