# House Price Prediction – MLOps Project

## Project Overview
This project predicts residential home sales prices using Machine Learning, Scikit-Learn Pipelines, Pandera Data Validation, and MLflow experiment tracking/model registry practices based on the Ames Housing dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Random Forest
- Ridge & Lasso Regression
- Pandera
- Joblib
- MLflow
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
housepricepredict/
│
├── artifacts/
│   ├── actual_vs_predicted.png
│   ├── residuals_distribution.png
│   ├── feature_importance.png
│   ├── preprocessing_summary_report.json
│   ├── reproducibility_report.json
│   └── production_model_report.json
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── dataset_metadata.json
│       ├── X_train_final.npy
│       ├── X_test_final.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── X_submission_final.npy
│       └── test_ids.npy
├── logs/
├── models/
│   ├── scaler.pkl
│   ├── ohe.pkl
│   ├── preprocessor.pkl
│   ├── ridge_baseline.pkl
│   ├── random_forest_baseline.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
├── notebooks/
│   ├── project_implementation.ipynb
│   └── house_price_project_separated (1).ipynb
├── outputs/
│   ├── submission.csv
│   ├── high_residual_errors.csv
│   ├── overpredicted_cases.csv
│   └── underpredicted_cases.csv
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   ├── run_lab6_registry.py
│   └── validate_reproducibility.py
├── src/
│   ├── automate_lifecycle.py
│   ├── evaluate.py
│   ├── generate_registry_report.py
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── train.py
│   ├── train_mlflow.py
│   ├── train_registry.py
│   ├── validate_data.py
│   └── validate_outputs.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Workflow

1. **Data Validation**: Strict schema and constraint validation with Pandera (`src/validate_data.py`).
2. **Data Preprocessing & Cleaning**: Outlier removal, missing value imputation, log-target transformation (`src/preprocess.py`).
3. **Feature Engineering**: Total square footage, bathroom count, porch square footage, house age, interaction terms (`QualSF`).
4. **Pipeline Encapsulation**: Modular Scikit-Learn `ColumnTransformer` (`RobustScaler` + `OneHotEncoder`) (`src/preprocess_pipeline.py`).
5. **Output Validation**: Preprocessing verification, shape and NaN assertions (`src/validate_outputs.py`).
6. **Model Training**: Baseline (Ridge, Random Forest) and Advanced models (XGBoost, LightGBM) (`src/train.py`).
7. **Model Evaluation & Error Analysis**: Evaluation with log RMSE, MAE, R², Dollar errors, residual analysis, and Kaggle submission generation (`src/evaluate.py`).
8. **MLflow Tracking**: Hyperparameters, metrics, diagnostic plots (Actual vs Predicted, Residuals, Feature Importances) tracked in MLflow (`src/train_mlflow.py`).
9. **Model Registry & Lifecycle Automation**: Automated champion-challenger face-off and stage transitions (Staging -> Production -> Archived) (`src/train_registry.py`, `src/automate_lifecycle.py`).
10. **Deployment Artifacts**: JSON production specification report generation (`src/generate_registry_report.py`).

## Models

- **Baseline**: Ridge Regression & Random Forest Regressor
- **Production Champion**: Tuned XGBoost Regressor (lowest CV RMSE: ~0.118 – 0.124)

## Evaluation Metrics

- **Log RMSE**: Root Mean Squared Error on $\log(1 + \text{SalePrice})$ (Kaggle benchmark)
- **Log MAE**: Mean Absolute Error on log-scale
- **R² Score**: Coefficient of determination ($R^2 > 0.90$)
- **Dollar RMSE / MAE**: Real-world prediction error in USD ($)

## MLflow Tracking

MLflow is used to track:
- Parameters (regularization alpha, tree depth, estimators, learning rate)
- Metrics (`rmse`, `mae`, `r2_score`, `rmse_dollars`, `mae_dollars`)
- Model artifacts and lineage (`preprocessor.pkl`, model binaries)
- Diagnostic plots (`actual_vs_predicted.png`, `residuals_distribution.png`, `feature_importance.png`)
- Dataset metadata (`dataset_metadata.json`)

## How to Run

Activate your virtual environment and execute the modular scripts or full pipelines:

### 1. Lab 3: Baseline Pipeline
```bash
python pipelines/run_lab3_baseline.py
```
*(Runs preprocessing, baseline training, and evaluation)*

### 2. Lab 4: MLflow Tracking
```bash
python pipelines/run_lab4_tracking.py
```
*(Runs preprocessing and MLflow experiment runs)*

### 3. Lab 5: Production Data Pipeline
```bash
python pipelines/run_lab5_pipeline.py
```
*(Runs schema validation, ColumnTransformer pipeline, and output validation)*

### 4. Lab 6: Model Registry & Lifecycle
```bash
python pipelines/run_lab6_registry.py
```
*(Trains candidate, registers in MLflow registry, runs automated Champion vs Challenger promotion, and exports deployment JSON report)*

### 5. Reproducibility Test
```bash
python pipelines/validate_reproducibility.py
```
