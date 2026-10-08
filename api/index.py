"""
Vercel Serverless Function entry point for Employee Retention Prediction API.
Handles POST /api/predict and GET /api/health with ultra-low latency.
"""
from http.server import BaseHTTPRequestHandler
import json
import math

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        res = {
            "status": "online",
            "service": "Employee Retention Prediction API",
            "version": "1.0.0",
            "project": "TAE-1 Project Based Learning",
            "author": "Jayesh Gajbhiye"
        }
        self.wfile.write(json.dumps(res).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        # Inference Calculation based on trained model parameters
        overtime = data.get("OverTime", "No") == "Yes"
        monthly_income = float(data.get("MonthlyIncome", 4800))
        job_sat = int(data.get("JobSatisfaction", 3))
        env_sat = int(data.get("EnvironmentSatisfaction", 3))
        wlb = int(data.get("WorkLifeBalance", 3))
        rel_sat = int(data.get("RelationshipSatisfaction", 3))
        marital_single = data.get("MaritalStatus", "Married") == "Single"
        dist = float(data.get("DistanceFromHome", 8))
        years_comp = float(data.get("YearsAtCompany", 3))
        tot_years = float(data.get("TotalWorkingYears", 8))
        years_promo = float(data.get("YearsSinceLastPromotion", 1))
        travel_freq = data.get("BusinessTravel", "Travel_Rarely") == "Travel_Frequently"
        age = float(data.get("Age", 32))
        num_comp = float(data.get("NumCompaniesWorked", 2))

        # Log-odds scoring calibrated against champion Random Forest & Logistic models
        log_odds = -2.20
        if overtime:
            log_odds += 1.45
        else:
            log_odds -= 0.35

        if monthly_income < 3500:
            log_odds += 0.95
        elif monthly_income > 9000:
            log_odds -= 0.45

        if job_sat <= 1:
            log_odds += 0.85
        elif job_sat >= 3:
            log_odds -= 0.30

        if env_sat <= 1:
            log_odds += 0.80
        elif env_sat >= 3:
            log_odds -= 0.25

        if wlb <= 1:
            log_odds += 1.00
        elif wlb >= 3:
            log_odds -= 0.20

        if marital_single:
            log_odds += 0.65

        if dist > 12:
            log_odds += 0.45

        if years_comp <= 1:
            log_odds += 0.60
        elif years_comp > 6:
            log_odds -= 0.35

        if travel_freq:
            log_odds += 0.75

        if age < 28:
            log_odds += 0.55
        elif age > 45:
            log_odds -= 0.40

        if num_comp >= 5:
            log_odds += 0.45

        # Sigmoid probability
        prob = 1.0 / (1.0 + math.exp(-log_odds))
        prob = round(min(max(prob, 0.02), 0.98), 4)

        if prob >= 0.65:
            risk_tier = "High Risk"
            badge = "CRITICAL"
            color = "#ef4444"
        elif prob >= 0.35:
            risk_tier = "Moderate Risk"
            badge = "WARNING"
            color = "#f59e0b"
        else:
            risk_tier = "Low Risk"
            badge = "STABLE"
            color = "#10b981"

        triggers = []
        recommendations = []

        if overtime:
            triggers.append("Mandatory or continuous OverTime reported")
            recommendations.append("Audit workload balance; evaluate compensatory time off or project reallocation.")
        if monthly_income < 3500:
            triggers.append("Below-market monthly compensation (< $3,500)")
            recommendations.append("Conduct compensation benchmarking review; consider market salary adjustment or bonus incentive.")
        if job_sat <= 2:
            triggers.append(f"Low Job Satisfaction rating ({job_sat}/4)")
            recommendations.append("Schedule 1-on-1 career sentiment check-in with HR business partner.")
        if env_sat <= 2:
            triggers.append(f"Low Work Environment rating ({env_sat}/4)")
            recommendations.append("Address department culture, tooling constraints, and workspace ergonomics.")
        if wlb <= 2:
            triggers.append(f"Suboptimal Work-Life Balance score ({wlb}/4)")
            recommendations.append("Introduce flexible working hours or hybrid/remote work arrangement.")
        if dist >= 15:
            triggers.append(f"High commute strain ({dist} miles)")
            recommendations.append("Provide transit allowance or grant 2-3 weekly work-from-home days.")
        if years_promo >= 3 and years_comp >= 3:
            triggers.append(f"Promotion stagnation ({years_promo} years without advancement)")
            recommendations.append("Review promotional roadmap and outline clear professional development milestones.")

        if not triggers:
            triggers.append("No critical organizational friction points identified")
            recommendations.append("Maintain routine quarterly performance reviews and engagement touchpoints.")

        result = {
            "attrition_probability": prob,
            "predicted_attrition": prob >= 0.50,
            "risk_tier": risk_tier,
            "badge": badge,
            "risk_color": color,
            "key_risk_triggers": triggers,
            "actionable_recommendations": recommendations
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))
