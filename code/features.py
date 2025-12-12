# code/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureSpec:
    target: str = "price"
    numeric: Tuple[str, ...] = ("model_year", "milage")
    categorical: Tuple[str, ...] = (
        "brand", "model", "fuel_type", "engine", "transmission",
        "ext_col", "int_col", "accident", "clean_title"
    )


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(spec.numeric)),
            ("cat", cat_pipe, list(spec.categorical)),
        ],
        remainder="drop",
    )
    return preprocessor


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[spec.target])
    y = df[spec.target].to_numpy(dtype=float)
    return X, y
