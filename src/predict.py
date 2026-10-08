"""
Inference & Risk Scoring Engine for Employee Retention Prediction System.
Provides individual assessment, batch scoring, and automated HR retention recommendations.
"""
import os
import joblib
import pandas as pd
import numpy as np
from preprocess import engineer_features, DROP_COLS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

class RetentionPredictor:
    def __init__(self, model_name: str = "champion_model.joblib"):
        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        self.metadata_path = os.path.join(MODELS_DIR, "feature_metadata.joblib")
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.metadata = joblib.load(self.metadata_path)
        
    def _prepare_input(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        for col in DROP_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])
        if "Attrition" in df.columns:
            df = df.drop(columns=["Attrition"])
            
        df_feat = engineer_features(df)
        X_trans = self.preprocessor.transform(df_feat)
        return pd.DataFrame(X_trans, columns=self.metadata["feature_names"], index=df.index), df_feat

    def predict_single(self, employee_data: dict) -> dict:
        """Evaluates single employee and returns probability, risk tier, and tailored recommendations."""
        df_single = pd.DataFrame([employee_data])
        X_trans_df, df_feat = self._prepare_input(df_single)
        
        prob = float(self.model.predict_proba(X_trans_df)[0, 1])
        pred = int(prob >= 0.50)
        
        # Risk Categorization
        if prob >= 0.65:
            risk_tier = "High Risk"
            risk_color = "#e53e3e"
            badge = "CRITICAL"
        elif prob >= 0.35:
            risk_tier = "Moderate Risk"
            risk_color = "#dd6b20"
            badge = "WARNING"
        else:
            risk_tier = "Low Risk"
            risk_color = "#38a169"
            badge = "STABLE"
            
        # Personalized Risk Triggers & Interventions
        triggers = []
        recommendations = []
        
        if employee_data.get("OverTime") == "Yes":
            triggers.append("Mandatory or continuous OverTime reported")
            recommendations.append("Audit workload balance; evaluate compensatory time off or project reallocation.")
            
        if employee_data.get("MonthlyIncome", 10000) < 3500:
            triggers.append("Below-market monthly compensation (< $3,500)")
            recommendations.append("Conduct compensation benchmarking review; consider market salary adjustment or bonus incentive.")
            
        if employee_data.get("JobSatisfaction", 4) <= 2:
            triggers.append(f"Low Job Satisfaction rating ({employee_data.get('JobSatisfaction')}/4)")
            recommendations.append("Schedule 1-on-1 career sentiment check-in with HR business partner.")
            
        if employee_data.get("EnvironmentSatisfaction", 4) <= 2:
            triggers.append(f"Low Work Environment rating ({employee_data.get('EnvironmentSatisfaction')}/4)")
            recommendations.append("Address department culture, tooling constraints, and workspace ergonomics.")
            
        if employee_data.get("WorkLifeBalance", 4) <= 2:
            triggers.append(f"Suboptimal Work-Life Balance score ({employee_data.get('WorkLifeBalance')}/4)")
            recommendations.append("Introduce flexible working hours or hybrid/remote work arrangement.")
            
        if employee_data.get("YearsSinceLastPromotion", 0) >= 3 and employee_data.get("YearsAtCompany", 0) >= 3:
            triggers.append(f"Promotion stagnation ({employee_data.get('YearsSinceLastPromotion')} years without advancement)")
            recommendations.append("Review promotional roadmap and outline clear professional development milestones.")
            
        if employee_data.get("DistanceFromHome", 0) >= 15:
            triggers.append(f"High commute strain ({employee_data.get('DistanceFromHome')} miles)")
            recommendations.append("Provide transit allowance or grant 2-3 weekly work-from-home days.")
            
        if not triggers:
            triggers.append("No critical organizational friction points identified")
            recommendations.append("Maintain routine quarterly performance reviews and engagement touchpoints.")
            
        return {
            "attrition_probability": round(prob, 4),
            "predicted_attrition": bool(pred),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "badge": badge,
            "key_risk_triggers": triggers,
            "actionable_recommendations": recommendations
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Processes a full roster DataFrame and appends predictions and risk tiers."""
        X_trans_df, _ = self._prepare_input(df_batch)
        probs = self.model.predict_proba(X_trans_df)[:, 1]
        
        df_result = df_batch.copy()
        df_result["Attrition_Probability"] = np.round(probs, 4)
        df_result["Predicted_Attrition"] = np.where(probs >= 0.50, "Yes", "No")
        
        conditions = [
            df_result["Attrition_Probability"] >= 0.65,
            df_result["Attrition_Probability"] >= 0.35
        ]
        choices = ["High Risk", "Moderate Risk"]
        df_result["Risk_Tier"] = np.select(conditions, choices, default="Low Risk")
        return df_result

if __name__ == "__main__":
    predictor = RetentionPredictor()
    test_emp = {
        "Age": 29,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 500,
        "Department": "Sales",
        "DistanceFromHome": 18,
        "Education": 3,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 45,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2400,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": 4,
        "OverTime": "Yes",
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 4,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0
    }
    res = predictor.predict_single(test_emp)
    print("Test Employee Prediction:")
    print(res)
