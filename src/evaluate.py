"""
Model Evaluation Utilities
===========================
Centralised plotting and metric functions so every notebook uses
consistent, publication-quality visuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Style defaults ────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


# ── Single-model metrics ──────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, y_prob, model_name: str = "Model") -> Dict[str, float]:
    """Print and return key classification metrics."""
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\n{report}")

    return {"model": model_name, "roc_auc": auc, "avg_precision": ap}


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save: bool = True):
    """Normalised confusion matrix with raw counts in parentheses."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=np.array([
            [f"{cm_norm[i,j]:.1%}\n({cm[i,j]})" for j in range(2)]
            for i in range(2)
        ]),
        fmt="",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"confusion_{model_name.replace(' ', '_')}.png")
    plt.show()


def plot_roc_curves(models_dict: Dict[str, Any], X_test, y_test, save: bool = True):
    """Overlay ROC curves for multiple fitted models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    colors = sns.color_palette("tab10", len(models_dict))
    for (name, model), color in zip(models_dict.items(), colors):
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        RocCurveDisplay.from_predictions(
            y_test, proba, name=f"{name} (AUC={auc:.3f})", ax=ax, color=color
        )

    ax.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "roc_comparison.png")
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save: bool = True,
):
    """Horizontal bar chart of the top-N most important features."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Blues_r", top_n)
    bars = ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    if save:
        import re
        safe_title = re.sub(r'[^\w\-]', '_', title)
        fig.savefig(FIGURES_DIR / f"{safe_title}.png")
    plt.show()


def plot_cv_results(cv_results: Dict[str, List[float]], save: bool = True):
    """Box-plot comparing cross-validation ROC-AUC scores across models."""
    df = pd.DataFrame(cv_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(ax=ax, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C9BE8", alpha=0.6))
    ax.set_ylabel("ROC-AUC (5-fold CV)", fontweight="bold")
    ax.set_title("Cross-Validation Score Distribution", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "cv_comparison.png")
    plt.show()


# ── Business impact ───────────────────────────────────────────────────────────

def business_impact(
    y_true,
    y_pred,
    avg_monthly_revenue: float = 65.0,
    avg_retention_cost: float = 50.0,
    months_saved: int = 12,
) -> pd.DataFrame:
    """
    Quantify the financial value of the model vs. no intervention.

    Assumptions
    -----------
    TP: Correctly flag a churner → retain them → save 12 months of revenue
    FP: Offer deal to loyal customer → waste retention budget
    FN: Miss a churner → lose 12 months of revenue
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    revenue_saved = tp * months_saved * avg_monthly_revenue
    retention_spend = (tp + fp) * avg_retention_cost
    revenue_lost = fn * months_saved * avg_monthly_revenue
    net_value = revenue_saved - retention_spend
    baseline_loss = (tp + fn) * months_saved * avg_monthly_revenue

    results = pd.DataFrame({
        "Metric": [
            "True Positives (churners caught)",
            "False Positives (wasted interventions)",
            "False Negatives (missed churners)",
            "Revenue saved by retaining TP",
            "Retention cost for TP + FP",
            "Revenue lost from FN",
            "Net value from model",
            "Baseline loss — no model",
        ],
        "Value": [
            tp, fp, fn,
            f"${revenue_saved:,.0f}",
            f"${retention_spend:,.0f}",
            f"${revenue_lost:,.0f}",
            f"${net_value:,.0f}",
            f"${baseline_loss:,.0f}",
        ],
    })
    return results
