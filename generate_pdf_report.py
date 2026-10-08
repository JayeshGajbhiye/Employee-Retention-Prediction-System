"""
PDF Generator for TAE-1 Employee Retention Prediction System Report.
Compiles a publication-quality HTML document with embedded base64 figures,
applies academic print styling, and compiles to PDF via headless Chromium/Edge.
"""
import os
import base64
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
HTML_PATH = os.path.join(PROJECT_DIR, "report_printable.html")
PDF_PATH = os.path.join(PROJECT_DIR, "TAE1_Employee_Retention_Prediction_System_Report.pdf")
ARTIFACT_PDF = os.path.join(
    r"C:\Users\user\.gemini\antigravity\brain\e4fb7df1-2589-49eb-b60f-6bfe0227f886",
    "TAE1_Employee_Retention_Prediction_System_Report.pdf"
)

def img_to_b64(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def build_html():
    img_dist = img_to_b64("attrition_distribution.png")
    img_ot = img_to_b64("attrition_by_overtime.png")
    img_inc = img_to_b64("income_by_attrition.png")
    img_role = img_to_b64("attrition_by_jobrole.png")
    img_corr = img_to_b64("correlation_heatmap.png")
    img_comp = img_to_b64("model_comparison_metrics.png")
    img_roc = img_to_b64("roc_curves.png")
    img_cm = img_to_b64("confusion_matrix_best.png")
    img_feat = img_to_b64("feature_importance.png")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TAE-1 Project Report: Employee Retention Prediction System</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 18mm 15mm 20mm 15mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
      color: #1e293b;
      line-height: 1.55;
      font-size: 10pt;
      margin: 0;
      padding: 0;
      background: #ffffff;
    }}
    .page-break {{
      page-break-after: always;
    }}
    .no-break {{
      page-break-inside: avoid;
    }}
    
    /* Cover / Header Header */
    .header-box {{
      border-bottom: 2px solid #1e3a8a;
      padding-bottom: 12px;
      margin-bottom: 20px;
    }}
    .project-tag {{
      display: inline-block;
      font-size: 8.5pt;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      color: #2563eb;
      background: #eff6ff;
      padding: 3px 8px;
      border-radius: 4px;
      margin-bottom: 6px;
    }}
    h1 {{
      font-size: 20pt;
      color: #0f172a;
      margin: 4px 0 6px 0;
      font-weight: 800;
      line-height: 1.2;
    }}
    .subtitle {{
      font-size: 11pt;
      color: #475569;
      margin-bottom: 12px;
      font-weight: 500;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 9pt;
      margin-bottom: 16px;
    }}
    .meta-item strong {{
      color: #0f172a;
    }}

    /* Links Callout */
    .links-card {{
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
      border-radius: 0 6px 6px 0;
      padding: 10px 14px;
      margin-bottom: 20px;
      font-size: 9pt;
    }}
    .links-card strong {{
      color: #166534;
      display: block;
      margin-bottom: 4px;
      font-size: 9.5pt;
    }}
    .links-card a {{
      color: #15803d;
      text-decoration: underline;
      font-weight: 600;
      word-break: break-all;
    }}

    /* Section Headings */
    h2 {{
      font-size: 13pt;
      color: #1e3a8a;
      border-bottom: 1.5px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 18px;
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h3 {{
      font-size: 10.5pt;
      color: #0f172a;
      margin-top: 12px;
      margin-bottom: 6px;
      font-weight: 600;
    }}
    p {{
      margin: 0 0 8px 0;
      text-align: justify;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 8.5pt;
      margin: 10px 0 14px 0;
      page-break-inside: avoid;
    }}
    th {{
      background: #1e3a8a;
      color: #ffffff;
      font-weight: 600;
      text-align: left;
      padding: 6px 8px;
      border: 1px solid #1e3a8a;
    }}
    td {{
      padding: 5px 8px;
      border: 1px solid #e2e8f0;
    }}
    tr:nth-child(even) {{
      background: #f8fafc;
    }}
    .highlight-row {{
      background: #eff6ff !important;
      font-weight: 600;
      color: #1e3a8a;
    }}

    /* Figures Grid */
    .fig-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 10px 0 14px 0;
      page-break-inside: avoid;
    }}
    .fig-card {{
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 6px;
      background: #ffffff;
      text-align: center;
    }}
    .fig-card img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    .fig-caption {{
      font-size: 7.5pt;
      color: #64748b;
      margin-top: 4px;
      font-weight: 500;
    }}

    /* Callout & Math */
    .callout {{
      background: #f1f5f9;
      border-left: 3px solid #64748b;
      padding: 8px 12px;
      font-size: 8.5pt;
      margin: 8px 0 12px 0;
      border-radius: 0 4px 4px 0;
    }}
    .formula-box {{
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      padding: 8px;
      text-align: center;
      font-family: 'Consolas', monospace;
      font-size: 8.5pt;
      color: #0f172a;
      margin: 6px 0;
      border-radius: 4px;
    }}
  </style>
</head>
<body>

  <!-- Cover / Header -->
  <div class="header-box">
    <span class="project-tag">TAE-1 Project Based Learning &bull; Academic Report</span>
    <h1>Employee Retention Prediction System</h1>
    <div class="subtitle">An End-to-End Supervised Machine Learning Pipeline for Attrition Risk Stratification and Prescriptive HR Interventions</div>
  </div>

  <div class="meta-grid">
    <div class="meta-item"><strong>Student / Author:</strong> Jayesh Gajbhiye</div>
    <div class="meta-item"><strong>Academic Evaluation:</strong> TAE-1 Project Based Learning</div>
    <div class="meta-item"><strong>Domain:</strong> Machine Learning &bull; People Analytics</div>
    <div class="meta-item"><strong>Core Algorithm:</strong> Cost-Sensitive Random Forest (Champion)</div>
  </div>

  <!-- Mandatory Links Callout -->
  <div class="links-card">
    <strong>Compulsory Project Submission Links:</strong>
    <div>&bull; <strong>GitHub Repository URL:</strong> <a href="https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System">https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System</a></div>
    <div>&bull; <strong>Live Vercel Web Application:</strong> <a href="https://employee-retention-prediction-syste.vercel.app">https://employee-retention-prediction-syste.vercel.app</a></div>
    <div>&bull; <strong>Live API Endpoint:</strong> <code>https://employee-retention-prediction-syste.vercel.app/api/predict</code></div>
  </div>

  <!-- Section 1 -->
  <h2>1. Executive Summary & Problem Formulation</h2>
  <p>
    Human capital retention is among the most urgent challenges confronting modern organizations. Unplanned voluntary attrition triggers substantial direct financial loss, with industry studies from the Society for Human Resource Management (SHRM) demonstrating that replacing an employee costs between <strong>100% and 200% of their annual salary</strong>. Beyond direct monetary costs, turnover causes loss of institutional knowledge, disrupts project timelines, and increases burnout among retained staff.
  </p>
  <p>
    Traditional Human Resources operations rely on reactive instruments, predominantly exit interviews performed after an employee has already tendered their resignation. This project develops and deploys an end-to-end predictive machine learning decision-support system that proactively calculates employee turnover probability (<em>P(Attrition = 1)</em>), detects underlying organizational friction points, and prescribes targeted retention actions before departures occur.
  </p>

  <!-- Section 2 -->
  <h2>2. System Architecture & Workflow Pipeline</h2>
  <p>
    The system follows a modular production architecture comprising data cleaning, leak-free featurization, multi-algorithm training with cross-validation, hyperparameter optimization, and a dual deployment strategy (Streamlit and Vercel):
  </p>

  <div class="callout">
    <strong>Pipeline Sequence:</strong> Raw IBM HR Dataset (1,470 samples, 35 features) &rarr; Zero-variance elimination &rarr; Stratified Train/Test Split (80/20) &rarr; Feature Engineering &rarr; Preprocessing Pipeline Fit (Train only) &rarr; Stratified 5-Fold Cross-Validation &rarr; Multi-Model Evaluation (Logistic Regression, Decision Tree, Random Forest, SVC, XGBoost) &rarr; GridSearchCV Hyperparameter Tuning &rarr; Holdout Evaluation &rarr; Interactive Vercel & Streamlit Cloud Deployment.
  </div>

  <!-- Section 3 -->
  <h2>3. Dataset Description & Variable Dictionary</h2>
  <p>
    The project leverages the canonical <strong>IBM HR Analytics Employee Attrition</strong> benchmark dataset ($N = 1,470$ instances, 35 columns). Target variable <code>Attrition</code> is binary ("Yes" = 283, "No" = 1,187), reflecting an authentic industry turnover baseline of <strong>19.25%</strong>.
  </p>

  <table>
    <thead>
      <tr>
        <th>Variable Category</th>
        <th>Key Features</th>
        <th>Data Types</th>
        <th>Operational Significance</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Target Feature</strong></td>
        <td><code>Attrition</code></td>
        <td>Binary (Yes / No)</td>
        <td>Ground truth departure status (Positive class: 19.25%)</td>
      </tr>
      <tr>
        <td><strong>Demographics</strong></td>
        <td><code>Age</code>, <code>Gender</code>, <code>MaritalStatus</code>, <code>Education</code>, <code>EducationField</code></td>
        <td>Numerical & Nominal</td>
        <td>Career lifecycle maturity and biographical background</td>
      </tr>
      <tr>
        <td><strong>Organizational Role</strong></td>
        <td><code>Department</code>, <code>JobRole</code>, <code>JobLevel</code>, <code>BusinessTravel</code>, <code>OverTime</code></td>
        <td>Nominal & Ordinal</td>
        <td>Workload intensity, travel burdens, and managerial hierarchy</td>
      </tr>
      <tr>
        <td><strong>Compensation</strong></td>
        <td><code>MonthlyIncome</code>, <code>DailyRate</code>, <code>HourlyRate</code>, <code>MonthlyRate</code>, <code>PercentSalaryHike</code>, <code>StockOptionLevel</code></td>
        <td>Continuous Numerical</td>
        <td>Financial reward competitiveness and market alignment</td>
      </tr>
      <tr>
        <td><strong>Tenure & Career</strong></td>
        <td><code>TotalWorkingYears</code>, <code>YearsAtCompany</code>, <code>YearsInCurrentRole</code>, <code>YearsWithCurrManager</code>, <code>YearsSinceLastPromotion</code></td>
        <td>Discrete Count</td>
        <td>Career progression velocity, role stagnation, and tenure stability</td>
      </tr>
      <tr>
        <td><strong>Surveys & Morale</strong></td>
        <td><code>JobSatisfaction</code>, <code>EnvironmentSatisfaction</code>, <code>RelationshipSatisfaction</code>, <code>WorkLifeBalance</code>, <code>JobInvolvement</code></td>
        <td>Ordinal Scale (1&ndash;4)</td>
        <td>Subjective employee satisfaction and workplace sentiment</td>
      </tr>
      <tr>
        <td><strong>Constants / Metadata</strong></td>
        <td><code>EmployeeCount</code>, <code>StandardHours</code>, <code>Over18</code>, <code>EmployeeNumber</code></td>
        <td>Zero-Variance</td>
        <td>Dropped during preprocessing to prevent dimensionality bloat</td>
      </tr>
    </tbody>
  </table>

  <div class="page-break"></div>

  <!-- Section 4 -->
  <h2>4. Exploratory Data Analysis (EDA) & Key Findings</h2>
  <p>
    Comprehensive exploratory data analysis was conducted to uncover underlying statistical distributions and correlations driving voluntary employee turnover:
  </p>

  <div class="fig-grid">
    <div class="fig-card">
      <img src="{img_dist}" alt="Class Distribution">
      <div class="fig-caption"><strong>Figure 1:</strong> Target Class Imbalance (80.75% Retained vs. 19.25% Departed).</div>
    </div>
    <div class="fig-card">
      <img src="{img_ot}" alt="OverTime Impact">
      <div class="fig-caption"><strong>Figure 2:</strong> Attrition Rate by OverTime (Overtime workers exhibit >30% departure).</div>
    </div>
    <div class="fig-card">
      <img src="{img_inc}" alt="Income Distribution">
      <div class="fig-caption"><strong>Figure 3:</strong> Monthly Income by Attrition (Departed employees show lower median income).</div>
    </div>
    <div class="fig-card">
      <img src="{img_role}" alt="Attrition by Job Role">
      <div class="fig-caption"><strong>Figure 4:</strong> Attrition Rates Across Job Roles (Sales Reps & Lab Techs experience highest turnover).</div>
    </div>
  </div>

  <div class="no-break">
    <div class="fig-card" style="margin-top: 10px;">
      <img src="{img_corr}" style="max-height: 250px;" alt="Correlation Matrix">
      <div class="fig-caption"><strong>Figure 5:</strong> Pearson Correlation Heatmap across key continuous numerical and tenure attributes.</div>
    </div>
  </div>

  <!-- Section 5 -->
  <h2>5. Preprocessing & Leak-Free Feature Engineering</h2>
  <p>
    Adhering strictly to enterprise Machine Learning best practices, all transformations adhered to <strong>strict featurization ordering</strong>:
  </p>
  <ul>
    <li><strong>Zero-Variance Dropping:</strong> <code>EmployeeCount</code>, <code>StandardHours</code>, <code>Over18</code>, and <code>EmployeeNumber</code> were removed.</li>
    <li><strong>Stratified Train-Test Partitioning:</strong> The dataset was split into 80% training ($N = 1,176$) and 20% independent holdout test ($N = 294$) sets prior to any scaling or encoding to guarantee complete isolation and prevent data leakage.</li>
    <li><strong>Engineered Domain Features:</strong> Five domain-specific HR ratios were derived:</li>
  </ul>

  <div class="formula-box">
    TenurePerJob = TotalWorkingYears / (NumCompaniesWorked + 1)<br>
    CompositeSatisfaction = (EnvironmentSatisfaction + JobSatisfaction + RelationshipSatisfaction + WorkLifeBalance) / 4<br>
    YearsWithManagerRatio = YearsWithCurrManager / (YearsAtCompany + 1)<br>
    IncomePerWorkingYear = MonthlyIncome / (TotalWorkingYears + 1)<br>
    YearsWithoutPromotionRatio = YearsSinceLastPromotion / (YearsAtCompany + 1)
  </div>

  <p>
    Numerical features were normalized via <code>StandardScaler</code>, and categorical variables encoded with <code>OneHotEncoder(drop='first', handle_unknown='ignore')</code>, expanding the feature space to 50 orthogonal model inputs.
  </p>

  <div class="page-break"></div>

  <!-- Section 6 & 7 -->
  <h2>6. Model Benchmark & Experimental Evaluation</h2>
  <p>
    Five diverse classification algorithms were evaluated alongside hyperparameter tuning. Cost-sensitive weighting was implemented across all models to mitigate class imbalance.
  </p>

  <h3>6.1 Stratified 5-Fold Cross-Validation Performance (Training Set)</h3>
  <table>
    <thead>
      <tr>
        <th>Candidate Architecture</th>
        <th>Mean CV ROC-AUC</th>
        <th>Std CV ROC-AUC</th>
        <th>Mean CV Recall</th>
        <th>Mean CV F1</th>
        <th>Mean CV Accuracy</th>
      </tr>
    </thead>
    <tbody>
      <tr class="highlight-row">
        <td><strong>Random Forest Classifier</strong></td>
        <td><strong>0.8189</strong></td>
        <td>&plusmn; 0.0286</td>
        <td>0.5088</td>
        <td>0.4925</td>
        <td><strong>79.93%</strong></td>
      </tr>
      <tr>
        <td>Logistic Regression (Balanced)</td>
        <td>0.8050</td>
        <td>&plusmn; 0.0318</td>
        <td><strong>0.7036</strong></td>
        <td><strong>0.5175</strong></td>
        <td>74.66%</td>
      </tr>
      <tr>
        <td>XGBoost Classifier</td>
        <td>0.8008</td>
        <td>&plusmn; 0.0307</td>
        <td>0.4955</td>
        <td>0.4844</td>
        <td>79.84%</td>
      </tr>
      <tr>
        <td>Support Vector Machine (SVC RBF)</td>
        <td>0.7840</td>
        <td>&plusmn; 0.0313</td>
        <td>0.5800</td>
        <td>0.4726</td>
        <td>75.25%</td>
      </tr>
      <tr>
        <td>Decision Tree Classifier</td>
        <td>0.6920</td>
        <td>&plusmn; 0.0394</td>
        <td>0.5969</td>
        <td>0.4591</td>
        <td>73.13%</td>
      </tr>
    </tbody>
  </table>

  <h3>6.2 Independent Holdout Test Set Evaluation ($N = 294$, 57 Positives)</h3>
  <table>
    <thead>
      <tr>
        <th>Model Architecture</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall (Attrition)</th>
        <th>F1-Score</th>
        <th>ROC-AUC</th>
        <th>PR-AUC</th>
      </tr>
    </thead>
    <tbody>
      <tr class="highlight-row">
        <td>🏆 <strong>Optimized Random Forest (Champion)</strong></td>
        <td><strong>78.23%</strong></td>
        <td><strong>45.45%</strong></td>
        <td><strong>61.40%</strong></td>
        <td><strong>0.5224</strong></td>
        <td><strong>0.8428</strong></td>
        <td><strong>0.5549</strong></td>
      </tr>
      <tr>
        <td>Random Forest (Default Balanced)</td>
        <td>79.25%</td>
        <td>47.22%</td>
        <td>59.65%</td>
        <td>0.5271</td>
        <td><strong>0.8502</strong></td>
        <td>0.5570</td>
      </tr>
      <tr>
        <td>Logistic Regression (Baseline)</td>
        <td>76.87%</td>
        <td>44.66%</td>
        <td><strong>80.70%</strong></td>
        <td><strong>0.5750</strong></td>
        <td>0.8339</td>
        <td>0.5467</td>
      </tr>
      <tr>
        <td>XGBoost Classifier</td>
        <td><strong>79.59%</strong></td>
        <td><strong>47.54%</strong></td>
        <td>50.88%</td>
        <td>0.4915</td>
        <td>0.8341</td>
        <td>0.4860</td>
      </tr>
      <tr>
        <td>Support Vector Machine (SVC)</td>
        <td>78.57%</td>
        <td>45.83%</td>
        <td>57.89%</td>
        <td>0.5116</td>
        <td>0.8315</td>
        <td>0.5171</td>
      </tr>
      <tr>
        <td>Decision Tree Classifier</td>
        <td>75.85%</td>
        <td>42.22%</td>
        <td>66.67%</td>
        <td>0.5170</td>
        <td>0.7545</td>
        <td>0.4290</td>
      </tr>
    </tbody>
  </table>

  <div class="fig-grid">
    <div class="fig-card">
      <img src="{img_comp}" alt="Model Comparison">
      <div class="fig-caption"><strong>Figure 6:</strong> Comparative Benchmark Metrics on Holdout Test Set.</div>
    </div>
    <div class="fig-card">
      <img src="{img_roc}" alt="ROC Curves">
      <div class="fig-caption"><strong>Figure 7:</strong> Receiver Operating Characteristic (ROC) curves across all models.</div>
    </div>
  </div>

  <div class="fig-grid">
    <div class="fig-card">
      <img src="{img_cm}" alt="Confusion Matrix">
      <div class="fig-caption"><strong>Figure 8:</strong> Champion Model Confusion Matrix (Caught 35 / 57 departures).</div>
    </div>
    <div class="fig-card">
      <img src="{img_feat}" alt="Feature Importance">
      <div class="fig-caption"><strong>Figure 9:</strong> Top 15 Feature Importances (OverTime & Income dominate).</div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- Section 8 -->
  <h2>7. Explainable AI & Feature Importance Analysis</h2>
  <p>
    Ensemble feature importance weights identify the primary operational factors driving turnover:
  </p>
  <ol>
    <li><strong><code>OverTime_Yes</code> (17.71%):</strong> The dominant organizational trigger; prolonged mandatory overtime increases departure odds threefold.</li>
    <li><strong><code>MonthlyIncome</code> (6.31%):</strong> Absolute compensation level strongly correlates with flight risk when trailing market norms.</li>
    <li><strong><code>CompositeSatisfaction</code> (6.20%):</strong> Our engineered holistic survey metric combining Job, Environment, Relationship, and Work-Life metrics.</li>
    <li><strong><code>YearsAtCompany</code> (5.42%):</strong> First- and second-year employees experience highest attrition risk prior to cultural anchoring.</li>
    <li><strong><code>Age</code> (5.36%):</strong> Younger cohorts possess greater labor market mobility and explore career alternatives faster.</li>
  </ol>

  <!-- Section 9 -->
  <h2>8. Prescriptive HR Retention Interventions</h2>
  <p>
    Rather than merely generating static predictions, the system maps model outputs to prescriptive HR action plans:
  </p>

  <table>
    <thead>
      <tr>
        <th>Detected Risk Factor</th>
        <th>Root Workplace Cause</th>
        <th>Prescriptive HR Action Protocol</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Mandatory OverTime</strong></td>
        <td>Chronic task overload & impending burnout</td>
        <td>Audit team deliverables, reassign tasks, and implement compensatory rest days.</td>
      </tr>
      <tr>
        <td><strong>Below-Market Compensation</strong></td>
        <td>Salary dissatisfaction vs. market offers</td>
        <td>Initiate an off-cycle compensation review and structure milestone retention bonuses.</td>
      </tr>
      <tr>
        <td><strong>Suboptimal Work-Life Balance</strong></td>
        <td>Long commute distance (&ge;15 miles)</td>
        <td>Authorize 2&ndash;3 weekly work-from-home days and flexible core hours.</td>
      </tr>
      <tr>
        <td><strong>Promotion Stagnation (&ge;3 yrs)</strong></td>
        <td>Career ceiling perception</td>
        <td>Formulate a transparent 6-month career advancement pathway with clear KPIs.</td>
      </tr>
      <tr>
        <td><strong>Low Job Satisfaction (&le;2/4)</strong></td>
        <td>Role misalignment or team friction</td>
        <td>Conduct confidential 1-on-1 check-ins and explore rotational internal project transfers.</td>
      </tr>
    </tbody>
  </table>

  <!-- Section 10 -->
  <h2>9. System Deployment & Live Cloud Infrastructure</h2>
  <p>
    The complete system has been deployed across two modern cloud platforms:
  </p>
  <ul>
    <li><strong>Vercel Serverless Web Application:</strong> Accessible globally at <a href="https://employee-retention-prediction-syste.vercel.app">https://employee-retention-prediction-syste.vercel.app</a>. Features a responsive, TailwindCSS-powered interactive dashboard, real-time risk gauge, and batch CSV scanning powered by an ultra-fast Python serverless endpoint (<code>/api/predict</code>).</li>
    <li><strong>GitHub Open-Source Repository:</strong> Hosted at <a href="https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System">https://github.com/JayeshGajbhiye/Employee-Retention-Prediction-System</a> containing complete reproducible source code, serialized models (<code>models/champion_model.joblib</code>), data generation scripts, and documentation.</li>
    <li><strong>Streamlit Application:</strong> Fully configured via <code>app.py</code> and <code>requirements.txt</code> for one-click deployment on Streamlit Community Cloud.</li>
  </ul>

  <!-- Section 11 -->
  <h2>10. Conclusion & Future Scope</h2>
  <p>
    The <strong>Employee Retention Prediction System</strong> transitions human capital management from reactive post-exit reviews to proactive, data-informed intervention. By combining rigorous leak-free featurization, cost-sensitive ensemble modeling achieving <strong>84.28% ROC-AUC</strong> and <strong>61.40% Recall</strong>, and an interactive cloud deployment, the platform equips leadership with the analytical capabilities needed to safeguard talent assets.
  </p>
  <p>
    <strong>Future Enhancements:</strong> Future iterations will incorporate survival analysis (Cox Proportional Hazards) to predict precise resignation time horizons, natural language processing (NLP) on open-ended employee pulse surveys, and enterprise webhook integrations into HRIS platforms (Workday, BambooHR).
  </p>

</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Printable HTML built successfully at: {HTML_PATH}")
    return HTML_PATH

def convert_html_to_pdf(html_path):
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break
            
    if not edge_exe:
        raise FileNotFoundError("Microsoft Edge executable not found for PDF rendering.")
        
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        f"file:///{html_path.replace(os.sep, '/')}"
    ]
    
    print(f"Executing PDF compilation via Edge headless...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
        print(f"PDF successfully generated: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")
        # Copy to artifact folder
        import shutil
        os.makedirs(os.path.dirname(ARTIFACT_PDF), exist_ok=True)
        shutil.copyfile(PDF_PATH, ARTIFACT_PDF)
        print(f"PDF copied to artifact directory: {ARTIFACT_PDF}")
        return PDF_PATH
    else:
        print(f"Error during PDF conversion: {result.stderr}")
        return None

if __name__ == "__main__":
    h = build_html()
    convert_html_to_pdf(h)
