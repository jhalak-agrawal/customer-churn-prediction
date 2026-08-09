"""
Data loading and cleaning for the Telco Customer Churn project.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/Telco-Customer-Churn.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has blank strings for customers with 0 tenure -> coerce & impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen is 0/1, make it Yes/No like the rest for consistency, then re-encode later
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop ID (not predictive)
    df = df.drop(columns=["customerID"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )

    # Count of subscribed add-on services (signal of engagement/stickiness)
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # Average monthly spend vs total charges / tenure (catches pricing changes / proration)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Charge per service - are they paying a lot for few services? (value-for-money signal)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    # Flag: no internet-based add-ons at all (higher churn risk group in EDA)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Flag: month-to-month + electronic check (known high-risk combo)
    df["high_risk_combo"] = (
        (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    return df


def build_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df = clean(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)
    print(df["Churn"].value_counts(normalize=True))
    df.to_csv("data/telco_clean.csv", index=False)
    print("Saved cleaned dataset to data/telco_clean.csv")
