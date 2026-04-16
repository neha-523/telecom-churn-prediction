# Telecom Churn Prediction

End-to-end ML project predicting customer churn using the real IBM Telco dataset (7,043 customers). Built as a data engineer expanding into ML — covers the full workflow from EDA to model explainability and business impact.

**Result:** LightGBM model achieving 0.85 ROC-AUC, catching 81% of churners before they leave (~$206K net value per cohort).

---

## Key findings

- Month-to-month customers churn at 43% vs 3% on 2-year contracts
- Customers under 12 months tenure are the highest risk segment
- Fiber optic users churn at 42% vs 19% for DSL
- OnlineSecurity and TechSupport add-ons meaningfully reduce churn

---

## Models compared (5-fold CV)

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.840 |
| Random Forest | 0.831 |
| XGBoost | 0.843 |
| **LightGBM (tuned)** | **0.848** |

---

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and save as `data/raw/telco_churn.csv`:
```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

```bash
python main.py          # full pipeline (~2 min)
python main.py --fast   # skip tuning
jupyter notebook        # explore notebooks
```

---

## Structure

```
├── notebooks/
│   ├── 01_eda.ipynb          exploratory data analysis
│   └── 02_modeling.ipynb     models, SHAP, business impact
├── src/
│   ├── preprocessing.py      sklearn pipeline + ColumnTransformer
│   ├── features.py           feature engineering
│   └── evaluate.py           metrics, plots, ROI
├── main.py                   end-to-end script
└── requirements.txt
```

---

**Stack:** Python · scikit-learn · LightGBM · XGBoost · SHAP · pandas · matplotlib · seaborn
