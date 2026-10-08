"""
Exploratory Data Analysis (EDA) module for Employee Retention Prediction System.
Generates statistical summaries and high-resolution visual plots for project documentation.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hr_employee_attrition.csv")

def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})
    
    df = pd.read_csv(DATA_PATH)
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    
    attrition_counts = df["Attrition"].value_counts()
    print(f"Attrition Distribution:\n{attrition_counts}\nPercentages:\n{df['Attrition'].value_counts(normalize=True)*100}")
    
    # 1. Attrition Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(attrition_counts.index, attrition_counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 15, f"{yval} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Employee Attrition Class Distribution (Target Variable)", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Number of Employees")
    ax.set_xlabel("Attrition Status")
    ax.set_ylim(0, max(attrition_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "attrition_distribution.png"), dpi=200)
    plt.close()
    
    # 2. Attrition by OverTime
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ot_df = df.groupby(["OverTime", "Attrition"]).size().unstack()
    ot_df_pct = ot_df.div(ot_df.sum(axis=1), axis=0) * 100
    ot_df_pct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d95f02"], ax=ax, edgecolor="black", linewidth=1.0)
    ax.set_title("Attrition Rate by OverTime Requirement", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Works OverTime")
    ax.set_xticklabels(["No Overtime", "Works Overtime"], rotation=0)
    ax.legend(title="Attrition", labels=["Stayed (No)", "Left (Yes)"])
    for n, c in enumerate(ot_df_pct.index):
        pos_y = 0
        for attr in ["No", "Yes"]:
            val = ot_df_pct.loc[c, attr]
            if val > 5:
                ax.text(n, pos_y + val / 2, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold")
            pos_y += val
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "attrition_by_overtime.png"), dpi=200)
    plt.close()
    
    # 3. Monthly Income vs Attrition (Boxplot & Violin Plot)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", palette=["#2b5c8f", "#d95f02"], ax=ax, width=0.4, boxprops=dict(alpha=0.85))
    ax.set_title("Monthly Income Distribution by Attrition Status", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Monthly Income (USD)")
    ax.set_xlabel("Attrition")
    median_no = df[df["Attrition"] == "No"]["MonthlyIncome"].median()
    median_yes = df[df["Attrition"] == "Yes"]["MonthlyIncome"].median()
    ax.text(0, median_no + 800, f"Median: ${median_no:,.0f}", ha="center", fontweight="bold", color="#1a365d")
    ax.text(1, median_yes + 800, f"Median: ${median_yes:,.0f}", ha="center", fontweight="bold", color="#7b241c")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "income_by_attrition.png"), dpi=200)
    plt.close()

    # 4. Job Role vs Attrition Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    role_pct = df.groupby("JobRole")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
    bars = ax.barh(role_pct.index, role_pct.values, color="#e6550d", edgecolor="black")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va="center", fontweight="bold")
    ax.set_title("Attrition Rate (%) by Job Role", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_xlim(0, max(role_pct.values) + 8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "attrition_by_jobrole.png"), dpi=200)
    plt.close()
    
    # 5. Correlation Heatmap of Key Numerical Features
    num_cols = [
        "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
        "YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion",
        "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome"
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix of Key Numerical Variables", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=200)
    plt.close()
    
    print(f"EDA visual plots successfully saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    run_eda()
