"""
Feature Engineering
====================
Domain-driven feature transformations applied BEFORE the sklearn preprocessor.
These are pandas-level transforms that create new columns.

Why do these before sklearn?
  - They're interpretable (you can explain them in interviews)
  - They use business domain knowledge, not just data statistics
  - They prevent leakage when computed on training set only where needed
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features. Returns a copy of df with new columns.

    New features
    ------------
    service_count       : number of add-on services subscribed (0-7)
                          More services → more locked-in, lower churn risk
    avg_monthly_per_yr  : MonthlyCharges / (tenure/12 + 1)
                          Normalises charges by customer age
    total_charges_log   : log1p of TotalCharges
                          Reduces right skew for linear models
    has_internet        : 1 if customer has any internet service
    is_autopay          : 1 if payment method is automatic (lower churn)
    tenure_bucket       : ordinal grouping of tenure into lifecycle stages
    """
    df = df.copy()

    # 1. Service count (internet add-ons + phone add-ons)
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = sum(
        (df[col] == "Yes").astype(int) for col in service_cols
    ) + (df["MultipleLines"] == "Yes").astype(int)

    # 2. Monthly charge normalised by tenure
    df["avg_monthly_per_yr"] = df["MonthlyCharges"] / (df["tenure"] / 12 + 1)

    # 3. Log of total charges
    df["total_charges_log"] = np.log1p(df["TotalCharges"].fillna(df["MonthlyCharges"]))

    # 4. Has internet
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # 5. Automatic payment
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # 6. Tenure bucket (customer lifecycle stage)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )

    return df


ENGINEERED_NUMERIC = [
    "service_count",
    "avg_monthly_per_yr",
    "total_charges_log",
    "has_internet",
    "is_autopay",
]

ENGINEERED_CAT = ["tenure_bucket"]
