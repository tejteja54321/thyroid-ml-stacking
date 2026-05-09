# Advanced Thyroid Disease Detection using Ensemble Machine Learning

## Overview

This project presents an advanced machine learning framework for multi-class thyroid disease classification using ensemble learning techniques. The system predicts:

- Hyperthyroid
- Hypothyroid
- Negative (Normal)

The proposed model combines multiple powerful machine learning algorithms using a Stacking Ensemble architecture to achieve highly accurate thyroid disease prediction.

---

## Features

- Advanced Data Preprocessing
- Missing Value Handling
- Label Encoding
- SMOTE Oversampling for Class Balancing
- SHAP Explainability
- Interactive Data Visualization
- Precision-Recall Analysis
- Ensemble Learning using Stacking
- Multi-Class Thyroid Classification

---

## Machine Learning Models Used

### Base Models
- LightGBM Classifier
- CatBoost Classifier
- XGBoost Classifier
- Random Forest Classifier

### Meta Model
- Logistic Regression

### Final Architecture
Stacking Classifier combining all base learners with Logistic Regression as meta learner.

---

## Dataset Features

The model uses the following medical attributes:

| Feature |
|---|
| age |
| sex |
| on_thyroxine |
| query_on_thyroxine |
| on_antithyroid_meds |
| sick |
| pregnant |
| thyroid_surgery |
| I131_treatment |
| query_hypothyroid |
| query_hyperthyroid |
| lithium |
| goitre |
| tumor |
| psych |
| TSH |
| T3 |
| TT4 |
| T4U |
| FTI |
| TBG |
| referral_source |

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Class Balancing using SMOTE
6. Model Training
7. Ensemble Stacking
8. Model Evaluation
9. SHAP Explainability
10. Prediction System

---

## Performance Metrics

| Metric | Value |
|---|---|
| Accuracy | 98.70% |
| Average Precision (Micro) | 99.92% |
| AU-PRC for Hyperthyroid | 97.00% |
| AU-PRC for Hypothyroid | 100.00% |
| AU-PRC for Negative | 100.00% |

---

## Final Model

The final model is a Stacking Ensemble consisting of:

- LightGBM
- CatBoost
- XGBoost
- Random Forest
- Logistic Regression (Meta Learner)

This architecture improved:
- Generalization
- Minority class detection
- Precision-Recall performance
- Overall robustness

---

## SHAP Explainability

SHAP was used to identify the most important medical features influencing thyroid disease prediction.

### Top Important Features
- TSH
- FTI
- TT4
- T3
- referral_source
- age
- TBG
- T4U
- on_thyroxine
- thyroid_surgery

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap imbalanced-learn
