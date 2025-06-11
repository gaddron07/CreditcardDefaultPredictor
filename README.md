# Credit Default Prediction 

## Overview

This project predicts whether a customer will default on their credit payment in the next month using advanced machine learning techniques. The primary goal is to maximize the recall for defaulters, which is crucial for financial institutions to reduce risk. We use the F2-score as the main metric (which weights recall higher than precision).

This repository is designed for full transparency and reproducibility — from raw data and preprocessing to model selection, explainability, and generating final predictions.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Step-by-Step Workflow](#step-by-step-workflow)
    1. [Business Understanding & Metric Selection](#1-business-understanding--metric-selection)
    2. [Data Acquisition & Exploration](#2-data-acquisition--exploration)
    3. [Data Preprocessing](#3-data-preprocessing)
    4. [Feature Engineering](#4-feature-engineering)
    5. [Handling Class Imbalance](#5-handling-class-imbalance)
    6. [Model Building & Experimentation](#6-model-building--experimentation)
    7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
    8. [Ensembling & Stacking](#8-ensembling--stacking)
    9. [Threshold Optimization](#9-threshold-optimization)
    10. [Evaluation & Model Selection](#10-evaluation--model-selection)
    11. [Model Explainability](#11-model-explainability)
    12. [Inference & Submission Preparation](#12-inference--submission-preparation)
    13. [Project Handover & Reproducibility](#13-project-handover--reproducibility)
    14. [Next Steps & Recommendations](#14-next-steps--recommendations)
3. [How to Run](#how-to-run)
4. [Requirements](#requirements)
5. [References & Acknowledgements](#references--acknowledgements)

---

## Project Structure

```
test/
├── Model_2_XGBoost_SMOTE_BayesOpt.ipynb
├── Model_3_Stacking_XGB_CatB_RF.ipynb
├── Model_5_CatBoost_SMOTE.ipynb
├── Model_6_Optuna_Stacking.ipynb
├── Model_7_Advanced_Stacking.ipynb
├── feature_importance_and_shap_all_models.py
├── train_dataset_final1.csv
├── validation.csv
├── inference_and_submission.py
├── final_high_f2_accuracy_model.joblib
├── imputer.joblib
├── scaler.joblib
├── requirements.txt
```
- **Notebooks:** Each notebook is a complete modeling experiment, from data prep to results.
- **Python Scripts:** For inference, feature importance, and SHAP explanations.
- **Artifacts:** Saved final model, imputer, and scaler for deployment.
- **Data:** Both training and validation datasets.

---

## Step-by-Step Workflow

### 1. Business Understanding & Metric Selection

- The project starts with the business need: predict next-month credit default to minimize risk.
- **Key metric:** F2-score (prioritizing recall on defaulters over precision).

---

### 2. Data Acquisition & Exploration

- Collected and inspected `train_dataset_final1.csv` (labeled) and `validation.csv` (unlabeled).
- Performed exploratory data analysis (EDA):
    - Checked class imbalance (defaults are minority).
    - Analyzed feature distributions, missing values, and outliers.

---

### 3. Data Preprocessing

- **Missing values:**  
    - Median imputation for robust models.
    - KNNImputer in advanced models for context-aware imputation.
- **Scaling:**  
    - StandardScaler applied to all numerical features.
- **Encoding:**  
    - Label encoding for categorical features as needed.

---

### 4. Feature Engineering

- Added domain-specific features (e.g., payment delay counts, payment-to-bill ratios) in some models.
- Ensured all relevant raw and engineered features are included for all models.

---

### 5. Handling Class Imbalance

- **SMOTE:** Used to generate synthetic minority class (defaulter) samples in Models 2, 3, 5, 6.
- **SMOTETomek:** Used in Model 7 for both over- and under-sampling, yielding cleaner training data.

---

### 6. Model Building & Experimentation

Multiple models were trained and compared, each in its own notebook:

- **Model 2:** XGBoost + SMOTE + Bayesian Optimization
- **Model 3:** Stacking (XGBoost, CatBoost, RandomForest, LR meta-learner)
- **Model 5:** CatBoost + SMOTE
- **Model 6:** Stacking with Optuna-tuned XGBoost, LightGBM, CatBoost
- **Model 7:** Advanced stacking of XGBoost, LightGBM, CatBoost, RandomForest (with KNNImputer, SMOTETomek, Optuna, LR meta-learner)

---

### 7. Hyperparameter Tuning

- **Bayesian Optimization:** Used in XGBoost model.
- **Optuna:** Automated, efficient search for best parameters in Models 6 and 7.

---

### 8. Ensembling & Stacking

- **Stacking:** Combined diverse base models (tree-based + meta-learner).
- **Cross-validation:** 5-fold stratified cross-validation for robust validation, especially for advanced stacks.

---

### 9. Threshold Optimization

- Instead of default 0.5, models were tuned to use custom probability thresholds that maximize F2-score (typically 0.12–0.37).
- Ensures highest recall for defaulters, as required by the business.

---

### 10. Evaluation & Model Selection

- **Metrics:** F2-score, recall, accuracy, confusion matrix, classification report.
- **Model comparison:** Comprehensive table in the report shows model names, pipelines, metrics, and best thresholds.
- **Best model:** Model 7 — advanced stacking ensemble with KNNImputer, SMOTETomek, and Optuna, achieving F2 ≈ 0.91 and accuracy ≈ 0.85.

---

### 11. Model Explainability

- **Feature Importance:**  
    - Generated plots for each base model (XGBoost, LightGBM, CatBoost, RandomForest).
    - Meta-learner coefficients plotted to show relative weight of each base model.
- **SHAP Explanations:**  
    - Provided global (summary) and local (force plot) explanations for key base models.
    - Demonstrated which features drive default risk for both the population and specific customers.
- **Interpretability section:** Included in the report for stakeholders/regulators.

---

### 12. Inference & Submission Preparation

- **Pipeline:**  
    - Preprocess new data using saved imputer and scaler.
    - Predict with the saved stacking model.
    - Apply the selected probability threshold.
- **Script:** `inference_and_submission.py` automates the above steps.
- **Output:**  
    - Submission file (`submission_23114017.csv`) with `Customer_ID` and `next_month_default`.

---

### 13. Project Handover & Reproducibility

- All code, data, artifacts, and documentation are provided for full reproducibility.
- Anyone can:
    - Clone the repo
    - Install dependencies from `requirements.txt`
    - Run any notebook or script from start to finish to reproduce results and outputs
- The detailed report and this README guide the user through every step.

---

### 14. Next Steps & Recommendations

- **Deploy Model 7** for production scoring.
- **Monitor performance** and retrain as new data arrives or if performance drifts.
- **Enhance interpretability** using SHAP/LIME for business and regulatory purposes.
- **Expand features** with external data sources for even better predictive power.
- **Consider API deployment** if real-time scoring is required.

---

## How to Run

### Step 1: Clone the Repository

```bash
git clone <repo_url>
cd <repo>/test
```

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 3: Run Notebooks

- Open any notebook (e.g., `Model_7_Advanced_Stacking.ipynb`) and run all cells to see each model’s full pipeline and evaluation.
- This will generate `submission_23114017.csv`.
- Check the generated PNG files for interpretability.
---

## Requirements

See `requirements.txt` for all dependencies (pandas, scikit-learn, imbalanced-learn, xgboost, catboost, lightgbm, optuna, shap, etc.).

---

## References & Acknowledgements

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- Credit dataset and business context inspired by standard credit scoring challenges.

---

**For any questions, suggestions, or issues, please contact the project owner or raise a GitHub issue.**

---
