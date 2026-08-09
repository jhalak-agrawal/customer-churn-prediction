# Resume Bullets & Interview Talking Points

## Resume bullets (pick 1–2)

> Built an end-to-end customer churn prediction system (XGBoost, SHAP) on 7,000+ telecom
> customer records, achieving 0.85 ROC-AUC and 75% recall on churners; deployed as an
> interactive Streamlit dashboard for per-customer risk scoring and explanation.

> Engineered 6 derived features (tenure buckets, service-count, high-risk contract/payment
> combinations) that improved model discrimination and surfaced actionable retention drivers.

> Translated churn model output into a business case: targeting the top 20% highest-risk
> customers projected $283K in net annual retained revenue at a 13.5x campaign ROI.

## Likely interview questions & how to answer

**"Why XGBoost over logistic regression if the AUC gain was small (0.845 → 0.848)?"**
Be honest: the gain was marginal on this dataset because the signal is mostly linear/monotonic
(tenure, contract type). XGBoost's real value here was capturing feature interactions
(e.g., contract type × payment method) and giving cleaner SHAP-based explanations. In a
resume/interview context, this is a good moment to show you don't over-claim results.

**"Why not just use accuracy?"**
73.5% of customers don't churn, so a model predicting "no churn" for everyone gets 73.5%
accuracy for free while being useless. ROC-AUC/PR-AUC evaluate ranking quality independent
of a single threshold, and I additionally tuned the decision threshold to optimize F1 on
the minority (churn) class since false negatives are costlier than false positives here.

**"How did you handle class imbalance?"**
`class_weight="balanced"` / `scale_pos_weight` rather than SMOTE, to avoid synthetic
sample artifacts on a modestly sized dataset (7K rows). Worth mentioning you tried
SMOTE and it didn't materially change results if you experiment further.

**"How would this work in production?"**
Feature store or streaming pipeline instead of a static CSV, scheduled retraining,
drift monitoring, and — critically — an A/B test to replace the assumed 35% retention
success rate with a measured one before trusting the ROI number.

**"What's a weakness of this project?"**
The business impact numbers use assumed retention cost/success rate, not real campaign
data — I flagged this explicitly rather than presenting it as ground truth. That's the
honest, defensible answer.
