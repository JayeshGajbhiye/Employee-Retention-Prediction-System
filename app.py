"""
Employee Retention Prediction System - Streamlit Web Application
An interactive AI-powered decision support platform for HR and people management.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import RetentionPredictor
import joblib

# Page configuration
st.set_page_config(
    page_title="Employee Retention Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .mod-risk {
        background-color: #fef3c7;
        border-left: 6px solid #f59e0b;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .low-risk {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor(model_file="champion_model.joblib"):
    return RetentionPredictor(model_name=model_file)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/human-resources.png", width=70)
st.sidebar.title("HR Retention AI")
st.sidebar.markdown("**TAE-1 Project Based Learning**")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Active Prediction Engine:",
    ["Optimized Random Forest (Champion)", "XGBoost Classifier", "Logistic Regression (Baseline)"]
)

model_file_map = {
    "Optimized Random Forest (Champion)": "champion_model.joblib",
    "XGBoost Classifier": "xgboost_model.joblib",
    "Logistic Regression (Baseline)": "logistic_model.joblib"
}

predictor = load_predictor(model_file_map[model_choice])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Project Resources")
st.sidebar.markdown("- **GitHub Repo:** [View Source Code](https://github.com/employee-retention-system/retention-prediction-ml)")
st.sidebar.markdown("- **Live Deployment:** [Streamlit Cloud](https://employee-retention-predictor.streamlit.app)")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: IBM HR Analytics Benchmark | Model: Supervised ML")

# Main Title
st.markdown('<div class="main-header">💼 Employee Retention Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Proactive attrition identification and AI-driven retention intervention strategy platform</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Individual Employee Assessment",
    "📊 Batch Roster Evaluation",
    "📈 Model Evaluation & Metrics",
    "🔍 Exploratory Data Insights",
    "📑 TAE-1 Project Documentation"
])

# ==========================================
# TAB 1: INDIVIDUAL EMPLOYEE ASSESSMENT
# ==========================================
with tab1:
    st.subheader("Employee Profile & Work Attributes")
    st.write("Input current employee metrics to evaluate turnover probability and receive customized retention recommendations.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics & Role")
        age = st.slider("Age", 18, 65, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        
        role_options = {
            "Sales": ["Sales Executive", "Sales Representative", "Manager"],
            "Human Resources": ["Human Resources", "Manager"],
            "Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director", "Manager"]
        }
        job_role = st.selectbox("Job Role", role_options[department])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        distance = st.slider("Distance From Home (Miles)", 1, 30, 8)

    with col2:
        st.markdown("#### 💰 Compensation & Experience")
        job_level = st.selectbox("Job Level (1 to 5)", [1, 2, 3, 4, 5], index=1)
        monthly_income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=25000, value=4800, step=200)
        daily_rate = st.slider("Daily Rate ($)", 100, 1500, 800)
        hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 65)
        monthly_rate = st.slider("Monthly Rate ($)", 2000, 27000, 14000)
        percent_hike = st.slider("Percent Salary Hike (%)", 10, 25, 14)
        stock_option = st.selectbox("Stock Option Level (0 to 3)", [0, 1, 2, 3], index=0)
        overtime = st.selectbox("Works OverTime?", ["No", "Yes"], index=1)

    with col3:
        st.markdown("#### 🌟 Sentiment & Tenure")
        job_sat = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=2, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        env_sat = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=2, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        wlb = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=2, format_func=lambda x: f"{x} - " + ["Bad", "Good", "Better", "Best"][x-1])
        job_inv = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        rel_sat = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        
        tot_years = st.slider("Total Working Years", 0, 40, 8)
        years_comp = st.slider("Years At Company", 0, 30, 3)
        years_role = st.slider("Years In Current Role", 0, 20, 2)
        years_manager = st.slider("Years With Current Manager", 0, 20, 2)
        years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        num_comp = st.slider("Number of Companies Worked", 0, 9, 2)
        training_times = st.slider("Training Sessions Attended Last Year", 0, 6, 2)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: ["1 - Below College", "2 - College", "3 - Bachelor", "4 - Master", "5 - Doctor"][x-1], index=2)
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        perf_rating = 4 if percent_hike >= 20 else 3

    st.markdown("---")
    
    if st.button("🚀 Analyze Retention Risk", type="primary", use_container_width=True):
        emp_dict = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": env_sat,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_inv,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_sat,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_comp,
            "OverTime": overtime,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": perf_rating,
            "RelationshipSatisfaction": rel_sat,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": tot_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": wlb,
            "YearsAtCompany": years_comp,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager
        }
        
        result = predictor.predict_single(emp_dict)
        prob = result["attrition_probability"]
        risk_tier = result["risk_tier"]
        badge = result["badge"]
        
        st.subheader("📋 Retention Risk Analysis Report")
        
        rcol1, rcol2, rcol3 = st.columns([1, 1, 1])
        with rcol1:
            st.metric("Attrition Probability", f"{prob*100:.1f}%")
        with rcol2:
            st.metric("Risk Classification", risk_tier)
        with rcol3:
            st.metric("Retention Status Flag", badge)
            
        # Risk progress bar
        st.progress(prob)
        
        if risk_tier == "High Risk":
            st.markdown(f"""
            <div class="high-risk">
                <h4>⚠️ CRITICAL RETENTION ALERT</h4>
                <p>This employee exhibits a <b>{prob*100:.1f}% turnover probability</b>. Immediate proactive HR intervention is recommended within 7–14 business days.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_tier == "Moderate Risk":
            st.markdown(f"""
            <div class="mod-risk">
                <h4>⚡ MODERATE ATTRITION RISK</h4>
                <p>Turnover probability is <b>{prob*100:.1f}%</b>. Employee displays some dissatisfaction or work pressure indicators. Schedule a check-in.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="low-risk">
                <h4>✅ LOW ATTRITION RISK</h4>
                <p>Turnover probability is <b>{prob*100:.1f}%</b>. Employee profile indicates strong organizational alignment and retention stability.</p>
            </div>
            """, unsafe_allow_html=True)
            
        col_risk_l, col_risk_r = st.columns(2)
        with col_risk_l:
            st.markdown("#### 🔍 Primary Organizational Triggers")
            for trig in result["key_risk_triggers"]:
                st.markdown(f"- 🔴 **{trig}**")
                
        with col_risk_r:
            st.markdown("#### 💡 Prescriptive HR Interventions")
            for rec in result["actionable_recommendations"]:
                st.markdown(f"- 🎯 {rec}")

# ==========================================
# TAB 2: BATCH ROSTER EVALUATION
# ==========================================
with tab2:
    st.subheader("Batch Workforce Retention Scoring")
    st.write("Upload a company CSV roster to perform automated organization-wide retention scanning.")
    
    sample_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hr_employee_attrition.csv")
    if os.path.exists(sample_csv_path):
        sample_df = pd.read_csv(sample_csv_path)
        csv_data = sample_df.head(50).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Benchmark Sample Roster (50 Employees CSV)",
            data=csv_data,
            file_name="sample_employee_roster.csv",
            mime="text/csv"
        )
        
    uploaded_file = st.file_uploader("Upload Employee Roster (.csv)", type=["csv"])
    
    if uploaded_file is not None or st.button("⚡ Evaluate Sample Benchmark Roster (First 50 Employees)"):
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
        else:
            df_batch = pd.read_csv(sample_csv_path).head(50)
            
        with st.spinner("Processing roster through machine learning pipeline..."):
            scored_df = predictor.predict_batch(df_batch)
            
        st.success(f"Successfully processed {len(scored_df)} employee profiles.")
        
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        high_risk_count = (scored_df["Risk_Tier"] == "High Risk").sum()
        mod_risk_count = (scored_df["Risk_Tier"] == "Moderate Risk").sum()
        low_risk_count = (scored_df["Risk_Tier"] == "Low Risk").sum()
        avg_prob = scored_df["Attrition_Probability"].mean() * 100
        
        m1.metric("Total Headcount Evaluated", len(scored_df))
        m2.metric("High Risk Staff", f"{high_risk_count} ({high_risk_count/len(scored_df):.1%})")
        m3.metric("Moderate Risk Staff", f"{mod_risk_count} ({mod_risk_count/len(scored_df):.1%})")
        m4.metric("Avg Attrition Likelihood", f"{avg_prob:.1f}%")
        
        # Risk Tier Filter
        tier_filter = st.multiselect(
            "Filter By Risk Category:",
            ["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"]
        )
        
        filtered_df = scored_df[scored_df["Risk_Tier"].isin(tier_filter)]
        
        display_cols = [
            "Age", "Department", "JobRole", "MonthlyIncome", "OverTime",
            "JobSatisfaction", "WorkLifeBalance", "Attrition_Probability", "Risk_Tier"
        ]
        available_display = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_display].sort_values(by="Attrition_Probability", ascending=False),
            use_container_width=True
        )
        
        # Download Scored Results
        out_csv = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Export Scored Retention Assessment (CSV)",
            data=out_csv,
            file_name="employee_retention_predictions.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 3: MODEL EVALUATION & METRICS
# ==========================================
with tab3:
    st.subheader("Model Performance & Experimental Benchmark")
    st.write("Rigorous model comparison on holdout test set with Stratified 5-Fold Cross Validation.")
    
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 📊 Comparative Algorithm Metrics")
        comp_img = os.path.join(figures_dir, "model_comparison_metrics.png")
        if os.path.exists(comp_img):
            st.image(comp_img, use_container_width=True)
            
    with col_m2:
        st.markdown("#### 🎯 Receiver Operating Characteristic (ROC)")
        roc_img = os.path.join(figures_dir, "roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)

    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.markdown("#### 🧩 Champion Model Confusion Matrix")
        cm_img = os.path.join(figures_dir, "confusion_matrix_best.png")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)
            
    with col_m4:
        st.markdown("#### 🌟 Top 15 Feature Importances")
        feat_img = os.path.join(figures_dir, "feature_importance.png")
        if os.path.exists(feat_img):
            st.image(feat_img, use_container_width=True)

# ==========================================
# TAB 4: EXPLORATORY DATA INSIGHTS
# ==========================================
with tab4:
    st.subheader("Workforce Behavioral Analytics & Patterns")
    st.write("Exploratory Data Analysis revealing the structural and workplace drivers of attrition.")
    
    c1, c2 = st.columns(2)
    with c1:
        img_ot = os.path.join(figures_dir, "attrition_by_overtime.png")
        if os.path.exists(img_ot):
            st.image(img_ot, caption="OverTime is the #1 single leading driver of attrition.", use_container_width=True)
            
    with c2:
        img_inc = os.path.join(figures_dir, "income_by_attrition.png")
        if os.path.exists(img_inc):
            st.image(img_inc, caption="Employees who depart have significantly lower median monthly income.", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        img_role = os.path.join(figures_dir, "attrition_by_jobrole.png")
        if os.path.exists(img_role):
            st.image(img_role, caption="Sales Representatives and Laboratory Technicians experience higher turnover.", use_container_width=True)
            
    with c4:
        img_corr = os.path.join(figures_dir, "correlation_heatmap.png")
        if os.path.exists(img_corr):
            st.image(img_corr, caption="Correlation matrix of key organizational metrics.", use_container_width=True)

# ==========================================
# TAB 5: TAE-1 PROJECT DOCUMENTATION
# ==========================================
with tab5:
    st.subheader("TAE-1 Project Based Learning Documentation")
    st.markdown("""
### 📌 Project Title
**Employee Retention Prediction System: A Machine Learning Approach to Identify Turnover Risk and Prescribe Retention Interventions**

### 🎯 Problem Statement
Employee turnover imposes severe financial costs (1.5x–2x annual salary per departed professional), operational disruption, knowledge drain, and declining team morale. Traditional human resources management relies on reactive exit interviews after the employee has already tendered their resignation. 

This project develops an end-to-end predictive machine learning system that identifies employees likely to leave an organization in advance, explains the primary drivers behind their turnover risk, and generates automated, actionable retention interventions.

---

### 🌐 Mandatory Links
- **GitHub Repository URL:**  
  [`https://github.com/employee-retention-system/retention-prediction-ml`](https://github.com/employee-retention-system/retention-prediction-ml)
- **Live Cloud Deployment URL:**  
  [`https://employee-retention-predictor.streamlit.app`](https://employee-retention-predictor.streamlit.app)

---

### 🏗️ Methodology & Technical Stack
1. **Data Ingestion & Integrity:** Canonical IBM HR Analytics Benchmark dataset (1,470 records, 35 attributes).
2. **Exploratory Data Analysis (EDA):** Multivariate distributions, correlation matrices, and class imbalance analysis (~19.2% positive attrition).
3. **Strict Featurization Ordering:** Split BEFORE fitting to prevent data leakage.
4. **Feature Engineering:** Domain-specific metrics including `TenurePerJob`, `CompositeSatisfaction`, `YearsWithManagerRatio`, and `IncomePerWorkingYear`.
5. **Class Imbalance Mitigation:** Integrated cost-sensitive class balancing (`class_weight='balanced'` and `scale_pos_weight`).
6. **Multi-Model Benchmark:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVC), and XGBoost.
7. **Evaluation:** Stratified 5-Fold Cross Validation, ROC-AUC, Recall, Precision, and PR-AUC.
8. **Deployment:** Interactive Streamlit web interface with real-time risk scoring and batch assessment.
    """)
