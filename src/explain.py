"""
SHAP explainability: global feature importance + save values for the app.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("models/churn_xgb_pipeline.joblib")
df = pd.read_csv("data/telco_clean.csv")
X = df.drop(columns=["Churn"])

preprocessor = model.named_steps["prep"]
clf = model.named_steps["clf"]

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

# Use a sample for speed
sample_idx = np.random.RandomState(42).choice(len(X), size=min(1500, len(X)), replace=False)
X_sample = X_transformed[sample_idx]

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)

# Global summary plot
plt.figure()
shap.summary_plot(
    shap_values, X_sample, feature_names=feature_names, show=False, max_display=15
)
plt.tight_layout()
plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight")
plt.close()

# Mean absolute SHAP -> top drivers table
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("reports/shap_feature_importance.csv", index=False)
print(importance_df.head(15).to_string(index=False))

print("\nSaved SHAP summary plot to reports/figures/shap_summary.png")
print("Saved feature importance table to reports/shap_feature_importance.csv")
