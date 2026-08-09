"""
Customer Churn Risk Dashboard
Run with: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_xgb_pipeline.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_clean.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
ref_df = load_reference_data()
preprocessor = model.named_steps["prep"]
clf = model.named_steps["clf"]
explainer = shap.TreeExplainer(clf)

st.title("📉 Customer Churn Risk Dashboard")
st.caption("XGBoost model · ROC-AUC 0.85 · trained on IBM Telco Customer Churn dataset")

tab1, tab2 = st.tabs(["🔎 Score a customer", "📊 Portfolio risk overview"])

# ---------------- TAB 1: single customer scoring ----------------
with tab1:
    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col3:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    if st.button("Predict churn risk", type="primary"):
        # Build a single-row input matching training schema, filling remaining
        # columns with the dataset mode/median as reasonable defaults
        row = ref_df.drop(columns=["Churn"]).iloc[[0]].copy()
        row["tenure"] = tenure
        row["Contract"] = contract
        row["MonthlyCharges"] = monthly_charges
        row["InternetService"] = internet_service
        row["PaymentMethod"] = payment_method
        row["PaperlessBilling"] = paperless_billing
        row["TechSupport"] = tech_support
        row["OnlineSecurity"] = online_security
        row["SeniorCitizen"] = senior
        row["Partner"] = partner
        row["Dependents"] = dependents
        row["MultipleLines"] = multiple_lines
        row["TotalCharges"] = monthly_charges * max(tenure, 1)
        row["avg_monthly_spend"] = row["TotalCharges"] / max(tenure, 1)
        row["has_internet"] = 0 if internet_service == "No" else 1
        row["high_risk_combo"] = 1 if (contract == "Month-to-month" and payment_method == "Electronic check") else 0
        service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        row["num_services"] = sum((row[c].iloc[0] == "Yes") for c in service_cols)
        row["charge_per_service"] = row["MonthlyCharges"] / (row["num_services"] + 1)
        bins = [-1, 6, 12, 24, 48, 72]
        labels = ["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"]
        row["tenure_bucket"] = pd.cut(row["tenure"], bins=bins, labels=labels)

        proba = model.predict_proba(row)[0, 1]

        risk_color = "🔴" if proba >= 0.5 else ("🟡" if proba >= 0.3 else "🟢")
        st.metric("Churn probability", f"{proba:.1%}", delta=None)
        st.markdown(f"### {risk_color} Risk level: {'High' if proba >= 0.5 else 'Medium' if proba >= 0.3 else 'Low'}")

        # SHAP explanation for this customer
        X_row_transformed = preprocessor.transform(row)
        feature_names = preprocessor.get_feature_names_out()
        shap_vals = explainer.shap_values(X_row_transformed)[0]

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:6]
        contrib_df = pd.DataFrame({
            "feature": [feature_names[i].replace("num__", "").replace("cat__", "") for i in top_idx],
            "impact": shap_vals[top_idx],
        }).sort_values("impact")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in contrib_df["impact"]]
        ax.barh(contrib_df["feature"], contrib_df["impact"], color=colors)
        ax.set_xlabel("Impact on churn risk (SHAP value)")
        ax.set_title("Why this prediction?")
        st.pyplot(fig)
        st.caption("🔴 Red = pushes risk up · 🔵 Blue = pushes risk down")

# ---------------- TAB 2: portfolio overview ----------------
with tab2:
    st.subheader("Portfolio-level risk overview")
    st.caption("Based on the held-out test set scored by the model")

    scored_path = os.path.join(os.path.dirname(__file__), "..", "reports", "test_set_scored.csv")
    if os.path.exists(scored_path):
        scored = pd.read_csv(scored_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers evaluated", f"{len(scored):,}")
        c2.metric("Predicted high risk (>50%)", f"{(scored['churn_proba'] >= 0.5).sum():,}")
        c3.metric("Avg predicted risk", f"{scored['churn_proba'].mean():.1%}")
        c4.metric("Actual churn rate", f"{scored['churn_actual'].mean():.1%}")

        st.markdown("#### Risk distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(scored["churn_proba"], bins=30, color="#4C72B0", alpha=0.8)
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)

        st.markdown("#### Top 20 highest-risk customers")
        top20 = scored.sort_values("churn_proba", ascending=False).head(20)
        display_cols = ["churn_proba", "tenure", "Contract", "MonthlyCharges", "InternetService", "PaymentMethod"]
        st.dataframe(top20[display_cols].style.format({"churn_proba": "{:.1%}", "MonthlyCharges": "${:.2f}"}))

        bi_path = os.path.join(os.path.dirname(__file__), "..", "reports", "business_impact.json")
        if os.path.exists(bi_path):
            import json
            with open(bi_path) as f:
                bi = json.load(f)
            st.markdown("#### Business impact (targeting top 20% risk)")
            b1, b2, b3 = st.columns(3)
            b1.metric("Revenue at risk (annual)", f"${bi['annual_revenue_at_risk_in_target_group']:,.0f}")
            b2.metric("Est. net impact", f"${bi['net_annual_impact']:,.0f}")
            b3.metric("Campaign ROI", f"{bi['roi_multiple']}x")
    else:
        st.warning("Run `python src/train.py` and `python src/business_impact.py` first to populate this tab.")

st.divider()
st.caption("Built with XGBoost + SHAP · Dataset: IBM Telco Customer Churn (7,043 customers)")
