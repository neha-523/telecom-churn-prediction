"""
main.py — End-to-End Churn Prediction Pipeline
================================================
Run the complete pipeline from raw data to saved model in one command:

    python main.py

Outputs
-------
  models/lgbm_final.pkl        Trained and tuned LightGBM pipeline
  reports/figures/             All evaluation charts
  reports/model_report.txt     Final metrics summary
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.preprocessing import load_raw, split_X_y, build_preprocessor, get_feature_names
from src.features import add_features
from src.evaluate import (
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_cv_results,
    business_impact,
    FIGURES_DIR,
)

DATA_PATH  = Path("data/raw/telco_churn.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(msg: str = ""):
    width = 60
    print(f"\n{'-'*width}")
    if msg:
        print(f"  {msg}")
        print(f"{'-'*width}")


def _check_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nData file not found: {DATA_PATH}\n"
            "Download the IBM Telco dataset:\n"
            "  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv\n"
            "and save it as data/raw/telco_churn.csv"
        )


# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_and_engineer(path: Path) -> tuple:
    """Load raw data and apply feature engineering."""
    df_raw = load_raw(path)
    print(f"Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns")

    df = add_features(df_raw)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"Missing values:\n{missing}\n→ Handled by median imputation in pipeline")

    X, y = split_X_y(df)
    return X, y


def train_and_tune(X_train, y_train) -> Pipeline:
    """Train LightGBM with randomised hyperparameter search."""
    param_dist = {
        "classifier__n_estimators":     [100, 200, 300, 500],
        "classifier__max_depth":        [3, 5, 7, -1],
        "classifier__num_leaves":       [20, 31, 50, 70],
        "classifier__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample":        [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_samples":[10, 20, 30],
        "classifier__reg_alpha":        [0, 0.01, 0.1],
        "classifier__reg_lambda":       [0, 0.01, 0.1, 1.0],
    }

    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LGBMClassifier(
            class_weight="balanced", random_state=SEED, verbose=-1
        )),
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest CV AUC : {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(model: Pipeline, X_test, y_test, feature_names: list):
    """Run full evaluation suite and write the report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = print_metrics(y_test, y_pred, y_prob, "LightGBM (tuned)")

    plot_confusion_matrix(y_test, y_pred, "LightGBM_tuned")
    plot_roc_curves({"LightGBM (tuned)": model}, X_test, y_test)

    # Feature importance from tree
    clf = model.named_steps["classifier"]
    importances = clf.feature_importances_
    plot_feature_importance(importances, feature_names, top_n=20,
                            title="LightGBM Feature Importance")

    impact = business_impact(y_test, y_pred)
    _sep("Business Impact")
    print(impact.to_string(index=False))

    # Write text report
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "model_report.txt"
    with open(report_path, "w") as f:
        f.write("TELECOM CHURN MODEL — FINAL REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ROC-AUC  : {metrics['roc_auc']:.4f}\n")
        f.write(f"Avg Prec : {metrics['avg_precision']:.4f}\n\n")
        f.write("BUSINESS IMPACT\n")
        f.write("-" * 40 + "\n")
        f.write(impact.to_string(index=False))
    print(f"\nReport saved → {report_path}")

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def run(skip_tuning: bool = False):
    start = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    _sep("1. Loading Data & Feature Engineering")
    _check_data()
    X, y = load_and_engineer(DATA_PATH)

    _sep("2. Train / Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Churn rate — Train: {y_train.mean():.1%}  Test: {y_test.mean():.1%}")

    if skip_tuning:
        _sep("3. Training LightGBM (no tuning)")
        model = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LGBMClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=SEED, verbose=-1
            )),
        ])
        model.fit(X_train, y_train)
    else:
        _sep("3. Hyperparameter Tuning (40 iterations × 5-fold CV)")
        model = train_and_tune(X_train, y_train)

    model_path = MODELS_DIR / "lgbm_final.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    _sep("4. Evaluation")
    feature_names = get_feature_names(model.named_steps["preprocessor"])
    evaluate(model, X_test, y_test, feature_names)

    elapsed = time.time() - start
    _sep(f"Done in {elapsed:.1f}s")
    print(f"Figures → {FIGURES_DIR}/")
    print(f"Model   → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction pipeline")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip hyperparameter tuning (faster run for testing)",
    )
    args = parser.parse_args()
    run(skip_tuning=args.fast)
