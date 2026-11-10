# An End-to-End MLOps Framework for House Price Prediction: Integrating Schema Validation, Experiment Tracking, Model Registry, and Containerized Serving

**Nidamanuru Shanmukha Ram**  
*Department of Artificial Intelligence and Machine Learning*  
*Vignan’s Foundation for Science, Technology and Research*  
*Bapatla, Andhra Pradesh, India*  
*Email: shanmukha.nidamanuru@gmail.com*  

---

***Abstract*—Accurate residential property valuation is essential for financial institutions, municipal taxing authorities, and prospective homebuyers. While traditional machine learning research has extensively examined algorithmic predictive accuracy, real-world deployment necessitates robust Machine Learning Operations (MLOps) architectures that guarantee data validity, experimental provenance, deterministic reproducibility, and automated microservice containerization. This paper presents an end-to-end, production-grade MLOps framework for residential house price estimation utilizing the 80-feature Ames Housing dataset. The proposed system incorporates domain-specific feature engineering, automated schema integrity validation via Pandera, multi-algorithm training across regularized linear models and gradient-boosted decision trees, centralized experiment tracking and model registration backed by MLflow, and containerized microservice serving via FastAPI, Docker, and Kubernetes. Seven regression architectures were evaluated under five-fold cross-validation. Tuned XGBoost demonstrated superior predictive accuracy, achieving a cross-validated Root Mean Squared Error (RMSE) of 0.11797 and a coefficient of determination ($R^2$) of 0.91215 on log-transformed prices. Furthermore, automated fault injection validated the system's schema guard resilience, while repeated pipeline executions verified 100% deterministic reproducibility. The resulting containerized microservice achieved low-latency inference ($<15$ ms) with an average prediction deviation below \$14,000, illustrating a scalable, robust paradigm for tabular enterprise MLOps.**

***Index Terms*—Ames housing dataset, containerized deployment, experiment tracking, feature engineering, machine learning operations (MLOps), model registry, predictive regression, schema validation, XGBoost.**

---

## I. Introduction

Machine learning (ML) has emerged as an indispensable methodology for modeling complex, non-linear socioeconomic interactions. In real estate analytics, property prices are dictated by a multitude of interacting attributes, ranging from structural dimensions and physical condition to locational proximity and neighborhood zoning [1]. Although predictive regression modeling on structured tabular datasets has achieved impressive empirical benchmarks, the discipline often overlooks the engineering challenges inherent in transitioning proof-of-concept models into resilient production systems.

In enterprise software engineering, raw machine learning models often suffer from the "silent failure" syndrome. Data corruptions—such as unexpected negative lot areas, unmapped categorical values, or shifted schema formats—rarely crash standard training pipelines; instead, they silently contaminate downstream feature transformations, yielding degraded predictions in production [2]. Moreover, the lack of rigorous model lineage, unversioned hyperparameter configurations, and manual deployment procedures severely undermine auditability and reproducibility.

To address these vulnerabilities, Machine Learning Operations (MLOps) establishes systematic engineering practices uniting machine learning system development and operations [3]. This research designs, implements, and evaluates an end-to-end MLOps pipeline for residential property valuation. Moving beyond monolithic script architectures, our framework establishes a modular, decoupled infrastructure encompassing automated data validation, Scikit-Learn pipeline encapsulation, MLflow lifecycle tracking, deterministic reproducibility validation, containerized REST API serving via FastAPI and Docker, and Kubernetes deployment orchestration.

---

## II. Problem Statement and Research Objectives

### A. Problem Statement
The primary analytical objective is to formulate a high-precision regression system capable of estimating the final transaction price (`SalePrice`) of single-family residences in Ames, Iowa, based on 80 heterogeneous attributes. Concurrently, the operational objective is to package this predictive system within a fault-tolerant, version-controlled MLOps lifecycle that ensures automated schema verification, traceable experiment logging, and microservice containerization.

### B. Research Objectives
This study addresses the following explicit research objectives:
1. **Exploratory Data Analysis & Preprocessing:** Systematically analyze feature distributions, identify and remove multivariate leverage outliers, and impute missing values based on structural domain logic.
2. **Domain-Specific Feature Engineering:** Formulate aggregate spatial, chronological, and interaction features that enhance tabular representation without introducing data leakage.
3. **Automated Schema Integrity Guard:** Construct a runtime schema verification layer using Pandera to enforce strict categorical domain boundaries and numerical bounds prior to pipeline execution.
4. **Comparative Algorithmic Benchmarking:** Develop, cross-validate, and tune seven diverse regression architectures: Ridge Regression, Lasso Regression, Elastic Net, Random Forest Regressor, Gradient Boosting Regressor, LightGBM Regressor, and XGBoost Regressor.
5. **Centralized Lifecycle Management:** Implement an SQLite-backed MLflow experiment tracking and model registry server to record hyperparameters, evaluation metrics, residual diagnostic plots, and artifact lineage.
6. **Containerized Production Serving:** Encapsulate the trained pipeline within a high-throughput FastAPI REST microservice packaged into a lightweight, standalone Docker container with Kubernetes orchestration descriptors.
7. **Reliability and Fault Resilience Auditing:** Conduct empirical fault-injection simulations and dual-execution reproducibility tests to verify deterministic pipeline stability.

---

## III. Related Work and Research Gap

### A. Classical and Ensemble Regression in Real Estate
Early automated valuation models (AVMs) relied predominantly on Ordinary Least Squares (OLS) and Hedonic Price Modeling [4]. While interpretable, linear specifications struggle to capture non-linear interactions, diminishing marginal returns of square footage, and localized spatial effects. Regularized formulations—notably Ridge ($L_2$) and Lasso ($L_1$) regression—introduced penalty constraints that mitigated multicollinearity and performed automated feature shrinkage [5]. 

The advent of tree-based ensemble learning shifted the paradigm. Breiman’s Random Forests [6] combined bootstrap aggregation with randomized subspace splitting, offering exceptional robustness against overfitting. Subsequently, gradient boosted decision trees (GBDT), pioneered by Friedman [7] and refined through XGBoost [8] and LightGBM [9], demonstrated empirical dominance across structured tabular benchmarks due to exact greedy split finding, second-order Taylor expansions, and cache-aware histogram optimizations.

### B. The MLOps Paradigm and Systemic Research Gaps
Despite mature algorithmic modeling, academic literature frequently divorces machine learning theory from operational sustainability. Sculley et al. [2] highlighted that core ML code represents only a fraction of real-world production ecosystems, with data verification, configuration management, and serving infrastructure consuming the vast majority of technical debt. 

Existing real estate regression studies predominantly publish static notebook experiments without automated data guards, reproducible pipeline orchestrators, or microservice packaging. This work bridges this gap by delivering a fully integrated, multi-stage MLOps ecosystem applied to tabular housing regression, serving as a blueprint for auditable, resilient enterprise AI deployments.

---

## IV. Dataset Architecture and System Workflow

### A. Dataset Characteristics
The project utilizes the benchmark Ames Housing Dataset compiled by Dean De Cock [10], which details residential real estate transactions in Ames, Iowa between 2006 and 2010. The dataset contains 1,460 training records across 81 columns (80 predictors and one continuous target: `SalePrice`) and 1,459 unseen test instances. The predictor variables encompass:
* **23 Nominal Categorical Variables:** e.g., `Neighborhood`, `BldgType`, `MSZoning`, `SaleCondition`.
* **23 Ordinal Categorical Variables:** e.g., `ExterQual`, `BsmtQual`, `KitchenQual`, `HeatingQC`.
* **14 Discrete Numerical Variables:** e.g., `YearBuilt`, `YearRemodAdd`, `FullBath`, `Fireplaces`.
* **20 Continuous Numerical Variables:** e.g., `LotArea`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`.

### B. High-Level System Workflow
The MLOps architectural workflow is structured into nine decoupled, modular phases as illustrated in the system flow:

$$\text{Raw Data Ingestion} \longrightarrow \text{Pandera Schema Guard} \longrightarrow \text{Domain Feature Engineering} \longrightarrow \text{Scikit-Learn Pipeline}$$
$$\longrightarrow \text{5-Fold Cross-Validation} \longrightarrow \text{MLflow Tracking \& Registry} \longrightarrow \text{Quality Gate Evaluation} \longrightarrow \text{Docker / FastAPI Serving}$$

---

## V. Preprocessing, Domain Engineering, and Schema Guard

### A. Outlier Treatment and Target Normalization
Preliminary exploratory analysis identified extreme influential outliers. In accordance with De Cock's recommendations [10], two properties with ground living areas exceeding 4,000 square feet but selling for unusually low prices ($< \$300,000$) were identified as atypical non-arm’s-length transactions and excised, yielding 1,458 training samples.

The target variable, `SalePrice`, exhibited positive skewness ($\gamma_1 = 1.88$) and kurtosis. To stabilize variance and satisfy normality assumptions for linear estimators, a logarithmic transformation was applied:

$$y_{\text{transformed}} = \log(1 + \text{SalePrice}) = \ln(1 + y)$$

Model predictions are dynamically inverted to the actual currency scale via the inverse exponential function during inference:

$$\hat{y}_{\text{currency}} = \exp(\hat{y}) - 1$$

### B. Domain-Specific Feature Engineering
To maximize the predictive capacity of raw features, domain-informed aggregates were engineered:
1. **Total Living Area (`TotalSF`):** Aggregate finished square footage:
   $$\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$$
2. **Total Bathroom Count (`Total_Bathrooms`):** Weighted fractional sum of full and half baths:
   $$\text{Total\_Bathrooms} = \text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$$
3. **Total Porch Area (`TotalPorchSF`):** Combined outdoor living space:
   $$\text{TotalPorchSF} = \text{WoodDeckSF} + \text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch}$$
4. **Temporal Decay Metrics:** `HouseAge` ($\text{YrSold} - \text{YearBuilt}$) and `RemodAge` ($\text{YrSold} - \text{YearRemodAdd}$).
5. **Quality-Area Interaction (`QualSF`):** High-leverage compound indicator:
   $$\text{QualSF} = \text{OverallQual} \times \text{TotalSF}$$
6. **Binary Facility Indicators:** Explicit flags denoting the presence of a pool, basement, second floor, garage, or fireplace.

### C. Pipeline Encapsulation and Automated Schema Guard
To eliminate data leakage between train and test splits, transformations were bundled inside a Scikit-Learn `ColumnTransformer`:
* **Numerical Pipeline:** Missing continuous values were imputed using neighborhood-grouped medians or constants ($0$), followed by `RobustScaler` to attenuate the influence of residual extreme values.
* **Categorical Pipeline:** Missing values were imputed with structural semantic categories (`None`), followed by sparse `OneHotEncoder(drop='first', handle_unknown='ignore')`.

The transformed feature matrix expands to **279 dimensionality columns**. To safeguard the pipeline against silent data corruption, a runtime Pandera schema guard was instituted in `src/validate_data.py`. The schema enforces strict typing, non-negative dimensions ($\text{LotArea} \ge 0, \text{GrLivArea} \ge 0$), and closed quality intervals ($\text{OverallQual} \in [1, 10]$).

---

## VI. Model Development, Tuning, and MLflow Tracking

### A. Algorithmic Formulation
Seven candidate regression algorithms spanning distinct functional families were implemented:
1. **Ridge Regression:** Penalized $L_2$ linear estimator minimizing $\|y - Xw\|_2^2 + \alpha \|w\|_2^2$.
2. **Lasso Regression:** Sparse $L_1$ penalized estimator minimizing $\|y - Xw\|_2^2 + \lambda \|w\|_1$.
3. **Elastic Net:** Balanced formulation combining $L_1$ and $L_2$ penalties with mixing parameter $\rho = 0.5$.
4. **Random Forest Regressor:** Bootstrap ensemble of 200 de-correlated decision trees optimizing mean squared error.
5. **Gradient Boosting Regressor:** Sequential additive ensemble fitting regression trees to pseudo-residuals.
6. **LightGBM Regressor:** Highly optimized tree booster employing Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).
7. **XGBoost Regressor:** Scalable boosting system incorporating second-order Hessian approximations, column subsampling, and shrinkage.

### B. Experiment Tracking Architecture
All training experiments were orchestrated via `src/train_mlflow.py` and logged to an SQLite-backed tracking repository (`sqlite:///mlflow.db`). For every execution, MLflow systematically captured:
* Full hyperparameter payloads (`n_estimators`, `max_depth`, `learning_rate`, `alpha`).
* Cross-validation metrics (RMSE, MAE, $R^2$, and Dollar-Scale RMSE).
* Serialization artifacts (`model.pkl`, `preprocessor.pkl`, Conda/pip dependency manifests).
* Diagnostic visualization artifacts, including predicted-versus-actual scatterplots, residual distribution histograms, and feature importance rankings.

---

## VII. Quality Gates, Reproducibility, and Model Registry

### A. Automated Production Quality Gate
Prior to deployment promotion, models must pass an automated multi-criteria quality gate instituted in `src/generate_registry_report.py`:
1. **Accuracy Threshold:** Log-scale test $\text{RMSE} \le 0.135$ and $R^2 \ge 0.85$.
2. **Data Consistency:** Zero permissible missing values ($\text{NaNs}$) in post-transformation matrices (`src/validate_outputs.py`).
3. **Reproducibility Guarantee:** Re-running the pipeline from scratch must yield bitwise identical loss scores.

### B. Deterministic Reproducibility Verification
In accordance with mission-critical MLOps practices, `pipelines/validate_reproducibility.py` executes back-to-back independent model training runs under seed fixed conditions ($\text{seed} = 42$). In our experimental evaluation, Run 1 and Run 2 produced identical validation scores:

$$\text{Execution 1 RMSE} = 0.12582103 \equiv \text{Execution 2 RMSE} = 0.12582103$$

The validation protocol returned status `PASSED` (`is_strictly_reproducible: true`), permanently documented in `artifacts/reproducibility_report.json`.

### C. MLflow Model Registry Promotion
Upon passing all quality gates, the winning candidate is registered into the MLflow Model Registry as `Ames_Housing_Production_Model` and formally promoted from `Staging` to `Production` status. The production artifact bundle is automatically mirrored to the `deployment/` directory for microservice consumption.

---

## VIII. Containerized Microservice Architecture and API Serving

### A. FastAPI Serving Layer
To serve predictions in real time, a high-performance RESTful API was developed in `api/main.py` and `src/serve_api.py`. Built upon Starlette and Pydantic, the service exposes three primary endpoints:
* **`GET /health`:** Performs active readiness and liveness probing, validating that both the serialized preprocessor and regression model are loaded into memory.
* **`POST /predict`:** Accepts single-property JSON payloads, verifies data types against Pydantic schemas, applies runtime feature engineering, transforms features via the loaded `ColumnTransformer`, and returns predicted sale prices in both log and US dollar formats.
* **`POST /batch_predict`:** High-throughput endpoint handling concurrent bulk evaluations.

### B. Docker Containerization and Kubernetes Orchestration
The serving service was packaged using an enterprise-ready `Dockerfile` built upon `python:3.11-slim`:
* Container dependencies were strictly isolated within `requirements-api.txt` to minimize surface attack area and image weight.
* Unnecessary files (`.venv`, `notebooks`, raw training data, `.git`) were pruned via `.dockerignore`.
* The multi-container orchestrator (`docker-compose.yml`) maps internal port `8000` to the host system with automatic restart policies.
* Kubernetes manifests (`k8s/deployment.yaml` and `k8s/service.yaml`) define a resilient deployment pod with replica management and a NodePort service.

Local integration testing with `test_api.py` verified seamless end-to-end inference, returning a predicted valuation of **\$208,291.85** on a standard benchmark residence within **11.4 milliseconds**.

---

## IX. Experimental Results, Metrics, and Discussion

### A. Cross-Validation Algorithmic Comparison
Table I summarizes the five-fold cross-validation performance across the seven evaluated regression algorithms on the log-transformed `SalePrice`.

```
========================================================================
TABLE I. FIVE-FOLD CROSS-VALIDATION PERFORMANCE COMPARISON
========================================================================
Algorithm                 Mean RMSE       Std. Dev. (RMSE)    Rank
------------------------------------------------------------------------
XGBoost Regressor          0.11797             0.00483          1
Lasso Regression           0.11870             0.00915          2
Elastic Net                0.11870             0.00916          3
Ridge Regression           0.11925             0.00867          4
LightGBM Regressor         0.12324             0.00833          5
Gradient Boosting          0.12333             0.00692          6
Random Forest Regressor    0.13439             0.00772          7
========================================================================
```

XGBoost achieved the highest performance, registering the lowest mean RMSE of **0.11797** and the smallest cross-validation variance ($\sigma = 0.00483$), indicating superior generalization stability. Regularized linear estimators (Lasso, Elastic Net, Ridge) demonstrated competitive performance, indicating that extensive feature engineering combined with $L_1/L_2$ shrinkage successfully captured strong linear signals.

### B. Final Production Model Evaluation Metrics
Following hyperparameter optimization and tournament registration, the champion model was evaluated across standard error metrics as reported in Table II.

```
========================================================================
TABLE II. FINAL PRODUCTION MODEL PERFORMANCE METRICS
========================================================================
Metric                   Log-Scale Value          Currency Equivalent ($)
------------------------------------------------------------------------
Root Mean Squared Error      0.11797                     $13,617.59
Mean Absolute Error (MAE)    0.08081                     $9,412.30
Mean Squared Error (MSE)     0.01394                         --
Coefficient of Det. (R²)     0.91215                     0.91215
========================================================================
```

The coefficient of determination ($R^2 = 0.91215$) confirms that over 91.2% of property valuation variance is explained by the engineered feature representations. On the original financial scale, the average absolute prediction deviation was restricted to **\$9,412.30**, representing exceptional precision given an average housing valuation exceeding \$180,000.

### C. MLOps Automated Pipeline Verification
The complete end-to-end pipeline was executed via `pipelines/run_mlops_pipeline.py`. Table III outlines the verification status across all system stages.

```
========================================================================
TABLE III. END-TO-END MLOps PIPELINE VERIFICATION AUDIT
========================================================================
Stage  Pipeline Component            Target Script           Audit Status
------------------------------------------------------------------------
01     Data Preprocessing & Eng.     preprocess_pipeline.py    PASSED
02     Schema Integrity Guard        validate_data.py          PASSED
03     Model Training & MLflow       train_mlflow.py           PASSED
04     Evaluation & Error Auditing   evaluate.py               PASSED
05     Registry Quality Gate         generate_registry_report  PASSED
06     Deterministic Reproducibility validate_reproducibility  PASSED
07     In-Process & HTTP Serving     test_api.py               PASSED
08     Docker Image Packaging        run_lab11_docker.py       PASSED
========================================================================
```

### D. Feature Importance Analysis
Feature attribution extraction identified `QualSF` (Overall Quality $\times$ Total Living Area), `OverallQual`, `GrLivArea`, `TotalBsmtSF`, and `Neighborhood` as the primary drivers of residential real estate value. Properties located in affluent neighborhoods (e.g., Northridge, Stone Brook) commanded statistically significant premiums independent of square footage, confirming the necessity of categorical encoding in tabular regression.

---

## X. Limitations and Future Work

While the proposed framework demonstrates robust empirical performance, several domain limitations persist:
1. **Geographic Specificity:** The Ames dataset reflects localized Midwestern demographic and tax conditions. Transferability to dense metropolitan markets with multi-unit condominiums requires retraining on regional datasets.
2. **Temporal Market Drift:** Real estate valuations fluctuate with macroeconomic shifts, inflation, and mortgage interest rates, variables unobserved in static historical transaction logs.
3. **Future Research Directions:** Subsequent iterations will incorporate automated continuous monitoring for feature and concept drift using Evidently AI, asynchronous retraining triggers within Kubernetes cron-jobs, and explainable AI dashboards incorporating SHAP (SHapley Additive exPlanations) values directly into the FastAPI client interface.

---

## XI. Conclusion

This paper detailed the architecture, empirical evaluation, and operational deployment of an end-to-end MLOps pipeline for residential property valuation. By uniting domain-specific feature engineering, automated Pandera data guards, regularized and ensemble regression algorithms, centralized MLflow tracking, and containerized Docker serving, the framework resolves critical vulnerabilities associated with manual machine learning workflows. 

XGBoost was confirmed as the optimal predictive engine, attaining a cross-validated RMSE of 0.11797 and an $R^2$ of 0.91215, translating to an average dollar prediction error under \$10,000. The full pipeline achieved 100% deterministic reproducibility and sub-15ms REST API latency, delivering an auditable, enterprise-ready blueprint for operational machine learning in tabular regression domains.

---

## References

[1] D. De Cock, “Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project,” *Journal of Statistics Education*, vol. 19, no. 3, pp. 1–15, 2011.  
[2] D. Sculley et al., “Hidden technical debt in machine learning systems,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2015, pp. 2503–2511.  
[3] S. Alla and S. K. Adari, *Beginning MLOps with Python: Designing and Building Robust Data and Machine Learning Pipelines*. Berkeley, CA: Apress, 2021.  
[4] S. C. Bourassa, E. Cantoni, and M. Hoesli, “Predicting house prices with spatial hedonic models,” *Real Estate Economics*, vol. 38, no. 4, pp. 723–748, 2010.  
[5] R. Tibshirani, “Regression shrinkage and selection via the lasso,” *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 58, no. 1, pp. 267–288, 1996.  
[6] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.  
[7] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.  
[8] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.  
[9] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.  
[10] Kaggle, “House Prices: Advanced Regression Techniques,” 2016. [Online]. Available: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.  
[11] M. Zaharia et al., “Accelerating the machine learning lifecycle with MLflow,” *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39–45, 2018.  
[12] S. Ramirez, “FastAPI: Modern, fast (high-performance), web framework for building APIs with Python,” 2020. [Online]. Available: https://fastapi.tiangolo.com.  

---

## About the Author

**Nidamanuru Shanmukha Ram** is an Artificial Intelligence and Machine Learning scholar in the Department of Artificial Intelligence and Machine Learning at Vignan’s Foundation for Science, Technology and Research, Bapatla, Andhra Pradesh, India. His research interests encompass enterprise Machine Learning Operations (MLOps), predictive tabular regression, automated model governance, and scalable cloud containerization architectures.
