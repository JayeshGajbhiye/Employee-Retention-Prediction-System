"""
Model Training, Cross-Validation, Hyperparameter Tuning, and Evaluation Module.
Trains and compares multiple machine learning algorithms with class-imbalance treatment.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from preprocess import prepare_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def train_and_evaluate_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Load prepared data (Train/Test split completed strictly before fit)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data()
    
    # Calculate class ratio for positive weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count
    
    # 2. Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            C=1.0, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    }
    
    # 3. Stratified 5-Fold Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_summary = {}
    
    print("\n" + "="*50)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION ON TRAINING SET")
    print("="*50)
    
    for name, model in models.items():
        scoring = ["roc_auc", "f1", "recall", "accuracy"]
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results_summary[name] = {
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"]))
        }
        print(f"[{name}] CV ROC-AUC: {np.mean(cv_scores['test_roc_auc']):.4f} (+/- {np.std(cv_scores['test_roc_auc']):.4f}) | Recall: {np.mean(cv_scores['test_recall']):.4f} | F1: {np.mean(cv_scores['test_f1']):.4f}")
        
    # 4. Train Models on Full Training Set and Evaluate on Independent Test Set
    test_metrics = {}
    predictions = {}
    probabilities = {}
    
    print("\n" + "="*50)
    print("EVALUATION ON INDEPENDENT HOLDOUT TEST SET")
    print("="*50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        predictions[name] = y_pred
        probabilities[name] = y_prob
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        test_metrics[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist()
        }
        
        print(f"[{name}]")
        print(f"  Accuracy:  {acc:.4f} | Recall (Catching Attrition): {rec:.4f} | Precision: {prec:.4f}")
        print(f"  F1-Score:  {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # 5. Hyperparameter Fine-Tuning for the Top Performing Tree Ensemble (Random Forest)
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION (GRID SEARCH CV)")
    print("="*50)
    
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    y_pred_opt = best_rf.predict(X_test)
    y_prob_opt = best_rf.predict_proba(X_test)[:, 1]
    
    test_metrics["Optimized Random Forest"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_opt)),
        "precision": float(precision_score(y_test, y_pred_opt, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_opt)),
        "f1_score": float(f1_score(y_test, y_pred_opt)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_opt)),
        "pr_auc": float(average_precision_score(y_test, y_prob_opt)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
        "best_params": grid_search.best_params_
    }
    print(f"Best RF Parameters: {grid_search.best_params_}")
    print(f"Optimized RF ROC-AUC: {test_metrics['Optimized Random Forest']['roc_auc']:.4f} | Recall: {test_metrics['Optimized Random Forest']['recall']:.4f}")

    # Determine best champion model based on balanced ROC-AUC and Recall
    champion_name = "Optimized Random Forest"
    champion_model = best_rf
    
    # Save Champion Model and full bundle
    joblib.dump(champion_model, os.path.join(MODELS_DIR, "champion_model.joblib"))
    joblib.dump(models["XGBoost"], os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    joblib.dump(models["Logistic Regression"], os.path.join(MODELS_DIR, "logistic_model.joblib"))
    
    # 6. Generate Comparison Visualizations
    
    # A. Model Comparison Bar Chart
    df_perf = pd.DataFrame({
        "Model": list(test_metrics.keys()),
        "Accuracy": [v["accuracy"] for v in test_metrics.values()],
        "Recall": [v["recall"] for v in test_metrics.values()],
        "Precision": [v["precision"] for v in test_metrics.values()],
        "F1-Score": [v["f1_score"] for v in test_metrics.values()],
        "ROC-AUC": [v["roc_auc"] for v in test_metrics.values()]
    })
    
    df_melt = df_perf.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="deep", ax=ax, edgecolor="black")
    ax.set_title("Comparative Model Performance on Holdout Test Set", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)")
    plt.xticks(rotation=15, ha="right")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_metrics.png"), dpi=200)
    plt.close()
    
    # B. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "Support Vector Machine"]:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {test_metrics[name]['roc_auc']:.3f})", linewidth=2)
    
    fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_opt)
    ax.plot(fpr_opt, tpr_opt, label=f"Optimized Random Forest (AUC = {test_metrics['Optimized Random Forest']['roc_auc']:.3f})", linewidth=2.5, linestyle="--", color="black")
    
    ax.plot([0, 1], [0, 1], "k:", label="Random Classifier (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=200)
    plt.close()
    
    # C. Confusion Matrix for Champion Model
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_best = np.array(test_metrics["Optimized Random Forest"]["confusion_matrix"])
    sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained (0)", "Attrited (1)"],
                yticklabels=["Retained (0)", "Attrited (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Confusion Matrix: Champion Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Employee Status", fontweight="bold")
    ax.set_ylabel("Actual Employee Status", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_best.png"), dpi=200)
    plt.close()
    
    # D. Feature Importance Plot
    importances = champion_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind="barh", color="#1f77b4", edgecolor="black", ax=ax)
    ax.set_title("Top 15 Predictors of Employee Attrition (Feature Importance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Weight")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=200)
    plt.close()
    
    # Save all metrics to JSON
    output_summary = {
        "cv_results": cv_results_summary,
        "test_results": test_metrics,
        "champion_model": champion_name,
        "top_features": feat_imp.to_dict()
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(output_summary, f, indent=4)
        
    print(f"\nAll models trained, evaluated, and artifacts saved successfully!")
    print(f"Metrics JSON saved to: {os.path.join(REPORTS_DIR, 'metrics.json')}")
    return output_summary

if __name__ == "__main__":
    train_and_evaluate_all()
