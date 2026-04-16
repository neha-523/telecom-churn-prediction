# Data

## Dataset: IBM Telco Customer Churn

**Source:** IBM Sample Data Sets (publicly available)  
**Direct download URL:**
```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

**Instructions:**
1. Download the CSV from the URL above
2. Place it at: `data/raw/telco_churn.csv`

## Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes/No) |
| Churn rate | ~26.5% |
| Source | IBM Watson Analytics |

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| customerID | string | Unique customer identifier |
| gender | categorical | Male / Female |
| SeniorCitizen | int | 1 if customer is 65+ |
| Partner | categorical | Has a partner (Yes/No) |
| Dependents | categorical | Has dependents (Yes/No) |
| tenure | int | Months as customer (0–72) |
| PhoneService | categorical | Has phone service |
| MultipleLines | categorical | Has multiple phone lines |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity | categorical | Add-on: online security |
| OnlineBackup | categorical | Add-on: online backup |
| DeviceProtection | categorical | Add-on: device protection |
| TechSupport | categorical | Add-on: tech support |
| StreamingTV | categorical | Streams TV |
| StreamingMovies | categorical | Streams movies |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling | categorical | Enrolled in paperless billing |
| PaymentMethod | categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | float | Monthly bill amount ($) |
| TotalCharges | float | Total billed to date ($) |
| **Churn** | **categorical** | **Target: Yes = churned** |
