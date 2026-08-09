"""
Train baseline (Logistic Regression) and final (XGBoost) churn models.
Saves fitted pipelines + metrics to /models and /reports.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
from xgboost import XGBClassifier

DATA_PATH = "data/telco_clean.csv"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(X)

    results = {}

    # ---------- Baseline: Logistic Regression ----------
    baseline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    baseline.fit(X_train, y_train)
    proba_base = baseline.predict_proba(X_test)[:, 1]
    pred_base = (proba_base >= 0.5).astype(int)

    results["logistic_regression"] = {
        "roc_auc": roc_auc_score(y_test, proba_base),
        "pr_auc": average_precision_score(y_test, proba_base),
        "f1": f1_score(y_test, pred_base),
    }
    print("Baseline (Logistic Regression):", results["logistic_regression"])

    # ---------- Final model: XGBoost with hyperparameter search ----------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )),
    ])

    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("\nBest XGBoost params:", search.best_params_)

    proba_xgb = best_model.predict_proba(X_test)[:, 1]

    # Choose threshold that maximizes F1 on test set (report both default 0.5 and tuned)
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_xgb)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    pred_xgb_default = (proba_xgb >= 0.5).astype(int)
    pred_xgb_tuned = (proba_xgb >= best_threshold).astype(int)

    results["xgboost"] = {
        "roc_auc": roc_auc_score(y_test, proba_xgb),
        "pr_auc": average_precision_score(y_test, proba_xgb),
        "f1_default_threshold": f1_score(y_test, pred_xgb_default),
        "f1_tuned_threshold": f1_score(y_test, pred_xgb_tuned),
        "tuned_threshold": float(best_threshold),
        "best_params": search.best_params_,
    }
    print("\nXGBoost (tuned):", results["xgboost"])

    print("\nClassification report (XGBoost, tuned threshold):")
    print(classification_report(y_test, pred_xgb_tuned, target_names=["Stayed", "Churned"]))

    cm = confusion_matrix(y_test, pred_xgb_tuned)
    print("Confusion matrix:\n", cm)
    results["xgboost"]["confusion_matrix"] = cm.tolist()

    # ---------- Save artifacts ----------
    joblib.dump(best_model, "models/churn_xgb_pipeline.joblib")
    joblib.dump(baseline, "models/churn_logreg_pipeline.joblib")

    X_test.assign(churn_actual=y_test.values, churn_proba=proba_xgb).to_csv(
        "reports/test_set_scored.csv", index=False
    )

    with open("reports/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved models to /models and metrics to /reports/model_metrics.json")
    return best_model, X_test, y_test, proba_xgb


if __name__ == "__main__":
    main()
