# Customer Churn Prediction & Retention ROI

Predicting which telecom customers are likely to churn, explaining *why* with SHAP, and
translating those predictions into a dollar-denominated retention campaign recommendation.

**Dataset**: [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) — 7,043 customers, 21 raw features.

---

## TL;DR Results

| Model | ROC-AUC | PR-AUC | F1 (churn class) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.845 | 0.653 | 0.62 |
| **XGBoost (tuned)** | **0.848** | **0.662** | **0.65** |

- At a tuned decision threshold, the model catches **75% of actual churners** (recall) while keeping precision at 57% — a deliberate trade-off since missing a churner is costlier than a false alarm.
- Targeting the **top 20% highest-risk customers** with a retention offer captures **67% of true churners** in that group (precision@k), for a projected **$283K net annual impact** and **13.5x ROI** on the campaign (see [business impact assumptions](#business-impact-methodology) below).
- Top churn drivers (via SHAP): **contract type** (month-to-month = highest risk), **tenure** (new customers churn most), lack of **online security**/**tech support**, and **fiber optic** internet service.

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── Telco-Customer-Churn.csv     # raw data
│   └── telco_clean.csv              # cleaned + feature-engineered
├── src/
│   ├── data_prep.py                 # cleaning + feature engineering
│   ├── eda.py                       # generates EDA figures
│   ├── train.py                     # baseline + XGBoost training, hyperparameter search
│   ├── explain.py                   # SHAP global explainability
│   └── business_impact.py           # translates predictions into $ terms
├── app/
│   └── streamlit_app.py             # interactive dashboard (single-customer scoring + portfolio view)
├── models/                          # saved model pipelines (.joblib)
├── reports/
│   ├── figures/                     # EDA + SHAP charts
│   ├── model_metrics.json
│   ├── business_impact.json
│   └── test_set_scored.csv
├── run_all.sh / run_all.bat         # run the entire pipeline in one command
├── requirements.txt
└── RESUME_BULLETS.md                # resume bullets + interview talking points
```

## How to Run

### Option A — one command (recommended)

```bash
pip install -r requirements.txt

# Windows (Command Prompt / double-click the file):
run_all.bat

# Mac/Linux/Git Bash:
bash run_all.sh
```

This runs the full pipeline (clean data → EDA → train → SHAP → business impact) in order and
launches the dashboard automatically at the end.

### Option B — step by step

```bash
pip install -r requirements.txt

python src/data_prep.py       # 1. Clean data + engineer features
python src/eda.py             # 2. Generate EDA charts
python src/train.py           # 3. Train baseline + tuned XGBoost model
python src/explain.py         # 4. SHAP explainability
python src/business_impact.py # 5. Business impact calculation
streamlit run app/streamlit_app.py  # 6. Launch interactive dashboard
```

---

## Approach

### 1. Data Cleaning
- `TotalCharges` had blank strings for customers with zero tenure (new signups); imputed as `MonthlyCharges × tenure`.
- Target encoded (`Churn`: Yes→1, No→0), customer ID dropped (non-predictive identifier).

### 2. Feature Engineering
- **`tenure_bucket`**: categorical tenure bands (0–6mo, 7–12mo, 1–2yr, 2–4yr, 4–6yr) to capture non-linear tenure effects.
- **`num_services`**: count of subscribed add-on services (security, backup, tech support, streaming) — a proxy for engagement/stickiness.
- **`avg_monthly_spend`** and **`charge_per_service`**: spend efficiency signals.
- **`high_risk_combo`**: flag for the known high-risk combination of month-to-month contract + electronic check payment (53.7% churn rate vs. 16.8% baseline in EDA).

### 3. Handling Class Imbalance
Churn rate is ~26.5%. Addressed via `class_weight="balanced"` (logistic regression) and `scale_pos_weight` (XGBoost) rather than oversampling, to avoid synthetic-data artifacts in a relatively small dataset.

### 4. Model Selection
- **Baseline**: Logistic Regression — interpretable coefficients, fast, good sanity check.
- **Final model**: XGBoost, tuned via `RandomizedSearchCV` (10 iterations, 3-fold stratified CV — kept small so it trains in well under a minute on a laptop) over depth, learning rate, subsampling, and regularization parameters.
- **Why ROC-AUC and PR-AUC over accuracy**: with 73.5% of customers not churning, a model predicting "no churn" for everyone would be 73.5% accurate but useless. PR-AUC is more informative under class imbalance; ROC-AUC lets us compare ranking quality independent of threshold choice.
- **Threshold tuning**: rather than defaulting to 0.5, the decision threshold was chosen to maximize F1 on the churn class, since false negatives (missed churners) are more costly to the business than false positives (an unnecessary retention offer).

### 5. Explainability (SHAP)
Used `TreeExplainer` on the tuned XGBoost model to produce both global feature importance and per-customer explanations (visible in the dashboard's "why this prediction?" panel) — critical for a retention team to trust and act on the model's output, not just its score.

### 6. Business Impact Methodology
Assumptions (clearly separated from the model so they can be swapped for real company figures):
- Retention offer cost: **$15/customer** targeted
- Retention offer success rate: **35%** of true churners who receive an offer are retained
- Targeting strategy: **top 20%** of customers by predicted churn probability

With these assumptions, on the held-out test set (1,409 customers): targeting the top 20% (281 customers) costs $4,215 and is projected to retain ~66 customers who would otherwise have churned, saving ~$61K in annual recurring revenue — a **net impact of ~$57K and 13.5x ROI** on that slice. Extrapolated to the full 7,043-customer base: **~$283K net annual impact**.

*These are illustrative assumptions for a portfolio project — in a real deployment, retention cost and success rate would come from actual campaign data (or an A/B test).*

---

## Key EDA Findings

| Segment | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Fiber optic internet | 41.9% |
| DSL internet | 19.0% |
| No internet service | 7.4% |
| Month-to-month + electronic check (high-risk combo) | 53.7% |
| Everyone else | 16.8% |

Average tenure: **18.0 months** for churned customers vs. **37.6 months** for retained customers.

---

## What I'd Do Next (Production Considerations)
- Replace static CSV with a live feature store / streaming pipeline for real-time scoring.
- A/B test the retention campaign to replace assumed success-rate with measured lift.
- Monitor for feature drift (e.g., pricing changes, new contract types) and retrain on a schedule.
- Add fairness checks across demographic segments (gender, senior citizen status) to ensure the model isn't systematically over/under-flagging protected groups.
- Calibrate probabilities (Platt scaling / isotonic regression) if the raw scores are used for anything beyond ranking.

---

## Tech Stack
Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib/seaborn
