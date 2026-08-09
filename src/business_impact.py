"""
Translate model predictions into a business-facing revenue-at-risk /
retention-campaign ROI story. Assumptions are clearly flagged so they
can be swapped for real company numbers in an interview.
"""
import pandas as pd
import json

# ---- Assumptions (documented, easy to challenge in an interview) ----
RETENTION_OFFER_COST = 15          # $ cost per customer targeted (discount/incentive)
RETENTION_SUCCESS_RATE = 0.35      # probability a targeted at-risk customer is actually retained
TARGET_PERCENTILE = 0.20           # target the top 20% highest-risk customers

df = pd.read_csv("reports/test_set_scored.csv")

n_customers = len(df)
avg_monthly_value = df["MonthlyCharges"].mean()

# Revenue at risk = sum of monthly charges for customers predicted to churn (score-ranked)
df_sorted = df.sort_values("churn_proba", ascending=False).reset_index(drop=True)
n_target = int(len(df_sorted) * TARGET_PERCENTILE)
targeted = df_sorted.iloc[:n_target]

# Of the targeted group, how many are ACTUAL churners (this is where the model earns its keep)
actual_churners_in_target = targeted["churn_actual"].sum()
precision_at_k = actual_churners_in_target / n_target

# Annualized revenue at risk among the targeted group (12 months of monthly charges)
annual_revenue_at_risk = (targeted["MonthlyCharges"] * 12).sum()

# Campaign cost
campaign_cost = n_target * RETENTION_OFFER_COST

# Expected customers saved = successes among TRUE churners in the targeted group
# (retention offers only "save" someone who would have actually churned)
expected_saved = actual_churners_in_target * RETENTION_SUCCESS_RATE
expected_revenue_saved = expected_saved * targeted["MonthlyCharges"].mean() * 12

net_impact = expected_revenue_saved - campaign_cost
roi = (net_impact / campaign_cost) if campaign_cost > 0 else float("nan")

summary = {
    "test_set_customers": n_customers,
    "target_percentile": TARGET_PERCENTILE,
    "customers_targeted": n_target,
    "precision_at_k": round(float(precision_at_k), 3),
    "actual_churners_in_target": int(actual_churners_in_target),
    "annual_revenue_at_risk_in_target_group": round(float(annual_revenue_at_risk), 2),
    "campaign_cost": campaign_cost,
    "assumed_retention_success_rate": RETENTION_SUCCESS_RATE,
    "expected_customers_saved": round(float(expected_saved), 1),
    "expected_annual_revenue_saved": round(float(expected_revenue_saved), 2),
    "net_annual_impact": round(float(net_impact), 2),
    "roi_multiple": round(float(roi), 2),
}

with open("reports/business_impact.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

# Extrapolate the same precision/economics to the full customer base (7,043) for a
# resume-friendly headline number
FULL_BASE = 7043
scale = FULL_BASE / n_customers
print(f"\nExtrapolated to full customer base (~{FULL_BASE} customers, x{scale:.1f} scale):")
print(f"  Net annual impact: ${summary['net_annual_impact'] * scale:,.0f}")
