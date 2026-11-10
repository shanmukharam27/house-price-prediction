# Complete Execution Guide: Ames House Price Prediction (Labs 1 to 13)

**Project Location:** `C:\Users\shanm\OneDrive\Desktop\housepricepredict`  
**Dataset:** Ames Housing (Iowa) Dataset (Regression: `SalePrice`)  
**Scope:** Complete Step-by-Step Execution Guide for all 13 Labs (Mirrored from `churnpredict`)  

---

## Quick Navigation
* [Project Location Confirmation](#project-location-confirmation)
* [Prerequisites & Environment Activation](#prerequisites--environment-activation)
* [Master One-Command Execution](#master-one-command-execution)
* [Lab-by-Lab Detailed Execution (Labs 1–13)](#lab-by-lab-detailed-execution)
  * [Labs 1 to 4: Baseline Regression & MLflow Experiment Tracking](#labs-1-to-4-baseline-regression--mlflow-experiment-tracking)
  * [Lab 5: Modular Preprocessing Pipeline & Pandera Schema Validation](#lab-5-modular-preprocessing-pipeline--pandera-schema-validation)
  * [Lab 6: MLflow Model Registry & Promotion Tournament](#lab-6-mlflow-model-registry--promotion-tournament)
  * [Lab 7: Model Evaluation, Residual Analysis & Error Auditing](#lab-7-model-evaluation-residual-analysis--error-auditing)
  * [Lab 8: Continuous Integration with Automated Testing](#lab-8-continuous-integration-with-automated-testing)
  * [Lab 9: Failure Injection & Data Corruption Simulation](#lab-9-failure-injection--data-corruption-simulation)
  * [Lab 10: Model Serving & REST API Engineering (FastAPI)](#lab-10-model-serving--rest-api-engineering-fastapi)
  * [Lab 11: Containerization with Docker & Kubernetes](#lab-11-containerization-with-docker--kubernetes)
  * [Lab 12: Continuous Monitoring, Drift Analysis & Retraining](#lab-12-continuous-monitoring-drift-analysis--retraining)
  * [Lab 13: End-to-End System Integration](#lab-13-end-to-end-system-integration)
* [Master Command Cheatsheet](#master-command-cheatsheet)

---

## Project Location Confirmation

> [!IMPORTANT]
> **Project Directory on Desktop:**  
> **`C:\Users\shanm\OneDrive\Desktop\housepricepredict`**  
> This directory is a clean copy of `housepricepredict'` and a complete architectural mirror of `churnpredict`. All pipeline scripts, src modules, notebooks, API configs, and deployment descriptors are organized in place.

---

## Prerequisites & Environment Activation

Open PowerShell inside the project directory:

```powershell
cd C:\Users\shanm\OneDrive\Desktop\housepricepredict

# Activate Python virtual environment (if present)
.venv\Scripts\Activate.ps1
```

---

## Master One-Command Execution

To execute the entire end-to-end pipeline in one step:

```powershell
python pipelines/run_mlops_pipeline.py
```

Or run the step-by-step PowerShell orchestrator:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

---

## Lab-by-Lab Detailed Execution

### Labs 1 to 4: Baseline Regression & MLflow Experiment Tracking
Trains Ridge, Random Forest, Gradient Boosting, XGBoost, and LightGBM models, logging parameters, RMSE, MAE, R², and residual diagnostic plots to MLflow:

```powershell
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
```

### Lab 5: Modular Preprocessing Pipeline & Pandera Schema Validation
Executes the scikit-learn ColumnTransformer pipeline, robust scaling, one-hot encoding, and Pandera schema validation:

```powershell
python pipelines/run_lab5_pipeline.py
```

### Lab 6: MLflow Model Registry & Promotion Tournament
Registers candidate models in the MLflow Model Registry and promotes the best performer to Production stage:

```powershell
python pipelines/run_lab6_registry.py
```

### Lab 7: Model Evaluation, Residual Analysis & Error Auditing
Generates residual analysis, prediction error breakdowns, and test evaluation reports:

```powershell
python src/evaluate.py
python pipelines/validate_reproducibility.py
```

### Lab 8: Continuous Integration with Automated Testing
Runs automated regression pipeline tests:

```powershell
pytest tests/ -v
```

### Lab 9: Failure Injection & Data Corruption Simulation
Injects corruptions (negative lot area, out-of-range quality, missing required columns) to test validation resilience:

```powershell
python src/simulate_data_corruption.py
```

### Lab 10: Model Serving & REST API Engineering (FastAPI)
Starts the FastAPI prediction microservice locally on port 8000:

```powershell
python src/serve_api.py
```

Verify in another terminal:

```powershell
python test_api.py --http
```

### Lab 11: Containerization with Docker & Kubernetes
Automates Docker container image building and verification:

```powershell
python pipelines/run_lab11_docker.py
```

Or manually:

```powershell
docker build -t house-price-prediction:v1.0 .
docker run -d -p 8000:8000 --name house-price-api house-price-prediction:v1.0
```

### Lab 12: Continuous Monitoring, Drift Analysis & Retraining
Runs the modular MLOps monitoring and drift detection pipeline:

```powershell
python src/mlops_pipeline/12_drift_analysis.py
python src/mlops_pipeline/12_retrain_model.py
```

### Lab 13: End-to-End System Integration
Executes all stages sequentially and records the results:

```powershell
python pipelines/run_mlops_pipeline.py
```

---

## Master Command Cheatsheet

| Lab | Description | Command |
| :--- | :--- | :--- |
| **Lab 3** | Baseline Model Pipeline | `python pipelines/run_lab3_baseline.py` |
| **Lab 4** | MLflow Experiment Tracking | `python pipelines/run_lab4_tracking.py` |
| **Lab 5** | Preprocessing & Schema Validation | `python pipelines/run_lab5_pipeline.py` |
| **Lab 6** | MLflow Registry & Lifecycle | `python pipelines/run_lab6_registry.py` |
| **Lab 7** | Evaluation & Error Auditing | `python src/evaluate.py` |
| **Lab 8** | Reproducibility & Unit Tests | `python pipelines/validate_reproducibility.py` |
| **Lab 9** | Fault Injection & Schema Guard | `python src/simulate_data_corruption.py` |
| **Lab 10** | FastAPI Prediction Service | `python src/serve_api.py` |
| **Lab 11** | Docker Containerization | `python pipelines/run_lab11_docker.py` |
| **Lab 12** | Drift & Monitoring Suite | `python src/mlops_pipeline/12_drift_analysis.py` |
| **All** | Master Pipeline Orchestrator | `python pipelines/run_mlops_pipeline.py` |
