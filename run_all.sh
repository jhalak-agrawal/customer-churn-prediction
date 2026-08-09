#!/bin/bash
# Run the entire churn prediction pipeline end-to-end, then launch the dashboard.
# Usage: bash run_all.sh

set -e  # stop immediately if any step fails

echo "=================================================="
echo "STEP 1/5: Cleaning data + feature engineering"
echo "=================================================="
python src/data_prep.py

echo ""
echo "=================================================="
echo "STEP 2/5: Generating EDA charts"
echo "=================================================="
python src/eda.py

echo ""
echo "=================================================="
echo "STEP 3/5: Training models (baseline + XGBoost)"
echo "=================================================="
python src/train.py

echo ""
echo "=================================================="
echo "STEP 4/5: SHAP explainability"
echo "=================================================="
python src/explain.py

echo ""
echo "=================================================="
echo "STEP 5/5: Business impact calculation"
echo "=================================================="
python src/business_impact.py

echo ""
echo "=================================================="
echo "ALL DONE. Launching the dashboard..."
echo "=================================================="
streamlit run app/streamlit_app.py
