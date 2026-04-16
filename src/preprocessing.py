"""
Data Loading & Preprocessing
=============================
Handles raw-data ingestion, cleaning, and building a reusable sklearn Pipeline
that can be fit on training data and applied to held-out sets — preventing leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ── Column groups ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",   # already 0/1, kept as numeric below
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_CAT_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

TARGET = "Churn"
ID_COL = "customerID"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_raw(path: str | Path) -> pd.DataFrame:
    """Load CSV and perform initial type corrections."""
    df = pd.read_csv(path)

    # TotalCharges has stray whitespace for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

    # SeniorCitizen is already int; ensure numeric
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop the ID column and split features from target."""
    X = df.drop(columns=[ID_COL, TARGET])
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
    - Imputes & scales numeric features
    - Ordinal-encodes binary Yes/No features (1/0)
    - One-hot-encodes multi-class categoricals (drop='first' for VIF reduction)
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False)),
    ])

    multi_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, [c for c in BINARY_FEATURES if c != "SeniorCitizen"]),
            ("senior", "passthrough", ["SeniorCitizen"]),
            ("cat", multi_cat_pipe, MULTI_CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting the preprocessor."""
    return list(preprocessor.get_feature_names_out())
