# 💼 Employee Retention Prediction System
### *TAE-1 Project Based Learning — End-to-End Machine Learning Solution*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
High employee turnover severely damages organizational productivity, creates substantial replacement costs (averaging 150% to 200% of an employee's annual salary), and precipitates intellectual property drain. Traditional HR responses rely on post-departure exit interviews, which are entirely reactive.

This project implements an intelligent **Employee Retention Prediction System** powered by Machine Learning. It analyzes multifaceted organizational attributes (demographics, job satisfaction, compensation, overtime strain, and career progression) to identify employees at high risk of attrition before they decide to leave, while prescribing concrete, personalized HR retention interventions.

---

## 🌐 Compulsory Project Links
- **GitHub Repository URL:** [https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System](https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System)
- **Live Vercel Web Application:** [https://employee-retention-prediction-syste.vercel.app](https://employee-retention-prediction-syste.vercel.app)
- **Live Streamlit Cloud Application:** [https://employee-retention-predictor.streamlit.app](https://employee-retention-predictor.streamlit.app)

---

## 🏗️ System Architecture & Workflow

```
[Raw HR Dataset (1,470 Records, 35 Features)]
                      │
                      ▼
[Data Cleaning & Zero-Variance Feature Elimination]
                      │
                      ▼
[Strict Featurization Split (80% Train / 20% Test)]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
 [Feature Engineering]     [Holdout Test Set]
 - TenurePerJob                     │
 - CompositeSatisfaction            │
 - YearsWithManagerRatio            │
 - IncomePerWorkingYear             │
         │                         │
         ▼                         │
 [Preprocessing Pipeline]           │
 - One-Hot Encoding (Nominal)       │
 - Standard Scaling (Continuous)    │
         │                         │
         ▼                         │
 [Cost-Sensitive Multi-Model Suite] │
 - Logistic Regression (Baseline)   │
 - Decision Tree Classifier         │
 - Random Forest (Champion)        │
 - Support Vector Machine (SVC)     │
 - XGBoost Classifier               │
         │                         │
         ▼                         │
 [Hyperparameter Optimization (GridSearchCV)]
         │                         │
         ▼                         │
 [Evaluation & Validation] ◄───────┘
 - Stratified 5-Fold Cross Validation
 - ROC-AUC: 0.8502 | Recall: 61.4%
                      │
                      ▼
 [Streamlit Web Application & Deployment]
 - Individual Employee Scoring & Triggers
 - Batch CSV Roster Scanning
 - Prescriptive HR Retention Actions
```

---

## 📊 Experimental Results & Model Benchmark

Evaluated on an independent holdout test set of 294 employees with stratified class distributions:

| Model Algorithm | Accuracy | Precision | Recall (Attrition) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Optimized Random Forest (Champion)** | **78.2%** | **45.5%** | **61.4%** | **0.522** | **0.843** | **0.555** |
| Random Forest (Default Balanced) | 79.3% | 47.2% | 59.6% | 0.527 | **0.850** | **0.557** |
| Logistic Regression (Baseline) | 76.9% | 44.7% | **80.7%** | **0.575** | 0.834 | 0.547 |
| XGBoost Classifier | **79.6%** | **47.5%** | 50.9% | 0.492 | 0.834 | 0.486 |
| Support Vector Machine (RBF) | 78.6% | 45.8% | 57.9% | 0.512 | 0.832 | 0.517 |
| Decision Tree Classifier | 75.9% | 42.2% | 66.7% | 0.517 | 0.755 | 0.429 |

> **Key Finding:** In human capital retention, **Recall** is prioritized over pure accuracy because failing to identify an attriting employee (False Negative) costs the enterprise thousands of dollars, whereas a False Positive simply results in a supportive manager-employee 1-on-1 discussion.

---

## 🔍 Top Predictors of Turnover
1. **OverTime (Yes/No):** Represents 17.7% of total feature importance; workers logging consistent overtime exhibit a 3x higher departure rate.
2. **Monthly Income:** Below-market entry compensation strongly correlates with near-term resignation.
3. **Composite Satisfaction:** Cross-survey average of Job, Environment, Relationship, and Work-Life balance.
4. **Years At Company:** Tenure under 2 years reflects the highest turnover vulnerability period.
5. **Age & Experience:** Younger professionals (early career) change roles at higher frequencies.

---

## 🚀 Quickstart: Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/employee-retention-system/retention-prediction-ml.git
cd retention-prediction-ml
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
# Generate benchmark dataset
python data/generate_dataset.py

# Run Exploratory Data Analysis
python src/eda.py

# Train models and generate performance charts
python src/train.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Directory Structure

```
employee_retention_prediction/
├── data/
│   ├── generate_dataset.py       # Benchmark dataset generator / loader
│   └── hr_employee_attrition.csv # Benchmark HR dataset (1,470 rows)
├── models/
│   ├── champion_model.joblib     # Optimized Random Forest model
│   ├── xgboost_model.joblib      # Trained XGBoost classifier
│   ├── logistic_model.joblib     # Trained Logistic Regression baseline
│   ├── preprocessor.joblib       # Fitted ColumnTransformer pipeline
│   └── feature_metadata.joblib   # Serialized feature names and mappings
├── reports/
│   ├── figures/                  # High-resolution evaluation charts
│   │   ├── attrition_distribution.png
│   │   ├── attrition_by_overtime.png
│   │   ├── income_by_attrition.png
│   │   ├── attrition_by_jobrole.png
│   │   ├── correlation_heatmap.png
│   │   ├── model_comparison_metrics.png
│   │   ├── roc_curves.png
│   │   ├── confusion_matrix_best.png
│   │   └── feature_importance.png
│   └── metrics.json              # Full quantitative benchmark results
├── src/
│   ├── eda.py                    # Exploratory Data Analysis routines
│   ├── preprocess.py             # Leak-free data transformations
│   ├── train.py                  # Training, CV, grid search & evaluation
│   └── predict.py                # Inference and recommendation engine
├── app.py                        # Multi-tab Streamlit dashboard
├── requirements.txt              # Cloud deployment dependencies
├── .gitignore                    # Git tracking rules
└── README.md                     # Comprehensive project documentation
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
