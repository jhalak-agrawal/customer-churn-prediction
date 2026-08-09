@echo off
echo ==================================================
echo STEP 1/5: Cleaning data + feature engineering
echo ==================================================
python src\data_prep.py
if errorlevel 1 goto :error

echo.
echo ==================================================
echo STEP 2/5: Generating EDA charts
echo ==================================================
python src\eda.py
if errorlevel 1 goto :error

echo.
echo ==================================================
echo STEP 3/5: Training models (baseline + XGBoost)
echo ==================================================
python src\train.py
if errorlevel 1 goto :error

echo.
echo ==================================================
echo STEP 4/5: SHAP explainability
echo ==================================================
python src\explain.py
if errorlevel 1 goto :error

echo.
echo ==================================================
echo STEP 5/5: Business impact calculation
echo ==================================================
python src\business_impact.py
if errorlevel 1 goto :error

echo.
echo ==================================================
echo ALL DONE. Launching the dashboard...
echo ==================================================
streamlit run app\streamlit_app.py
goto :eof

:error
echo.
echo Something failed above -- scroll up to see the error message.
pause
