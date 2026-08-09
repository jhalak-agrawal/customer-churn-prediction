"""
Exploratory Data Analysis - generates figures used in the report/README.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/telco_clean.csv")

FIG_DIR = "reports/figures"

# 1. Overall churn rate
plt.figure(figsize=(4, 4))
df["Churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot.pie(
    autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"], ylabel=""
)
plt.title("Overall Churn Rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_rate.png")
plt.close()

# 2. Churn rate by contract type
plt.figure(figsize=(6, 4))
rate = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
sns.barplot(x=rate.index, y=rate.values, color="#C44E52")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_contract.png")
plt.close()

# 3. Churn rate by tenure bucket
plt.figure(figsize=(6, 4))
order = ["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"]
rate = df.groupby("tenure_bucket", observed=True)["Churn"].mean().reindex(order) * 100
sns.barplot(x=rate.index, y=rate.values, color="#4C72B0")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Tenure")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_tenure.png")
plt.close()

# 4. Churn rate by internet service
plt.figure(figsize=(6, 4))
rate = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
sns.barplot(x=rate.index, y=rate.values, color="#DD8452")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Internet Service Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_internet.png")
plt.close()

# 5. Monthly charges distribution by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, alpha=0.4)
plt.title("Monthly Charges Distribution: Churned vs Retained")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/monthly_charges_dist.png")
plt.close()

# 6. Number of services vs churn
plt.figure(figsize=(6, 4))
rate = df.groupby("num_services")["Churn"].mean() * 100
sns.barplot(x=rate.index, y=rate.values, color="#55A868")
plt.xlabel("Number of Add-on Services")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate vs Number of Subscribed Services")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_num_services.png")
plt.close()

print("Saved 6 EDA figures to", FIG_DIR)

# Print key stats for README
print("\n--- Key EDA stats ---")
print("Churn by contract:\n", df.groupby("Contract")["Churn"].mean().round(3))
print("\nChurn by internet service:\n", df.groupby("InternetService")["Churn"].mean().round(3))
print("\nChurn by high_risk_combo flag:\n", df.groupby("high_risk_combo")["Churn"].mean().round(3))
print("\nAvg tenure churned vs retained:\n", df.groupby("Churn")["tenure"].mean().round(1))
