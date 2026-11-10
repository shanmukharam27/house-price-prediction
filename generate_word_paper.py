import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_ieee_paper():
    doc = Document()
    
    # Page setup - Section 1 (Title, Authors, Abstract - single column)
    section1 = doc.sections[0]
    section1.top_margin = Inches(1.0)
    section1.bottom_margin = Inches(1.125)
    section1.left_margin = Inches(0.75)
    section1.right_margin = Inches(0.75)
    section1.page_width = Inches(8.5)
    section1.page_height = Inches(11.0)
    
    # Configure default style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.0
    normal_style.paragraph_format.space_after = Pt(0)
    normal_style.paragraph_format.space_before = Pt(0)
    
    # 1. Main Title (24pt)
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(14)
    title_run = title_p.add_run("An End-to-End MLOps Framework for House Price Prediction: Integrating Schema Validation, Experiment Tracking, Model Registry, and Containerized Serving")
    title_run.font.name = 'Times New Roman'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    
    # 2. Author Block
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_p.paragraph_format.space_after = Pt(2)
    author_run = author_p.add_run("Nidamanuru Shanmukha Ram")
    author_run.font.name = 'Times New Roman'
    author_run.font.size = Pt(12)
    author_run.font.bold = True
    
    affil_p = doc.add_paragraph()
    affil_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affil_p.paragraph_format.space_after = Pt(2)
    affil_run = affil_p.add_run("Department of Artificial Intelligence and Machine Learning\nVignan's Foundation for Science, Technology and Research, Bapatla, Andhra Pradesh, India")
    affil_run.font.name = 'Times New Roman'
    affil_run.font.size = Pt(10)
    affil_run.font.italic = True
    
    email_p = doc.add_paragraph()
    email_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    email_p.paragraph_format.space_after = Pt(16)
    email_run = email_p.add_run("shanmukha.nidamanuru@gmail.com")
    email_run.font.name = 'Times New Roman'
    email_run.font.size = Pt(10)
    
    # 3. Abstract Block (single column, bold italic, 10pt)
    abstract_p = doc.add_paragraph()
    abstract_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abstract_p.paragraph_format.space_after = Pt(8)
    abstract_p.paragraph_format.left_indent = Inches(0.25)
    abstract_p.paragraph_format.right_indent = Inches(0.25)
    
    abs_title_run = abstract_p.add_run("Abstract-")
    abs_title_run.font.name = 'Times New Roman'
    abs_title_run.font.size = Pt(10)
    abs_title_run.font.bold = True
    abs_title_run.font.italic = True
    
    abs_text_run = abstract_p.add_run(" This paper presents an end-to-end, production-grade MLOps framework for residential house price estimation utilizing the 80-feature Ames Housing dataset. The proposed system incorporates domain-specific feature engineering, automated schema integrity validation via Pandera, multi-algorithm training across regularized linear models and gradient-boosted decision trees, centralized experiment tracking and model registration backed by MLflow, and containerized microservice serving via FastAPI, Docker, and Kubernetes. Seven regression architectures were evaluated under five-fold cross-validation. Tuned XGBoost demonstrated superior predictive accuracy, achieving a cross-validated Root Mean Squared Error (RMSE) of 0.11797 and a coefficient of determination (R2) of 0.91215 on log-transformed prices. Furthermore, automated fault injection validated the system's schema guard resilience, while repeated pipeline executions verified 100% deterministic reproducibility. The resulting containerized microservice achieved low-latency inference (<15 ms) with an average prediction deviation below $14,000, illustrating a scalable, robust paradigm for tabular enterprise MLOps.")
    abs_text_run.font.name = 'Times New Roman'
    abs_text_run.font.size = Pt(10)
    abs_text_run.font.bold = True
    abs_text_run.font.italic = True
    
    index_p = doc.add_paragraph()
    index_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    index_p.paragraph_format.space_after = Pt(18)
    index_p.paragraph_format.left_indent = Inches(0.25)
    index_p.paragraph_format.right_indent = Inches(0.25)
    
    idx_title_run = index_p.add_run("Index Terms-")
    idx_title_run.font.name = 'Times New Roman'
    idx_title_run.font.size = Pt(10)
    idx_title_run.font.bold = True
    idx_title_run.font.italic = True
    
    idx_text_run = index_p.add_run(" Ames housing dataset, containerized deployment, experiment tracking, feature engineering, machine learning operations (MLOps), model registry, predictive regression, schema validation, XGBoost.")
    idx_text_run.font.name = 'Times New Roman'
    idx_text_run.font.size = Pt(10)
    idx_text_run.font.bold = True
    idx_text_run.font.italic = True
    
    # 4. Section Break - Continuous to 2-Column Layout
    section2 = doc.add_section(WD_SECTION_START.CONTINUOUS)
    section2.top_margin = Inches(1.0)
    section2.bottom_margin = Inches(1.125)
    section2.left_margin = Inches(0.75)
    section2.right_margin = Inches(0.75)
    
    # Configure 2 columns (width 3.25 in, space 0.3125 in = 450 dxa)
    sectPr = section2._sectPr
    cols = parse_xml(r'<w:cols %s w:num="2" w:space="450"/>' % nsdecls('w'))
    sectPr.append(cols)
    
    def add_heading_1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.italic = True
        return p

    def add_body(text, indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.0
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.17)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        return p

    # Section I
    add_heading_1("I. INTRODUCTION")
    add_body("Machine learning (ML) provides computational methods for learning complex patterns from historical data and deploying them to estimate unobserved phenomena. In residential real estate, property values depend upon interacting dimensions, physical condition, spatial proximity, and neighborhood zoning [1]. While tabular regression models achieve impressive theoretical benchmarks, the discipline often overlooks the software engineering challenges required to transition proof-of-concept models into resilient production systems.")
    add_body("In enterprise settings, raw machine learning models often suffer from 'silent failures.' Data corruptions—such as negative lot areas, unmapped categorical values, or shifted schema formats—rarely crash standard training pipelines; instead, they silently contaminate downstream feature transformations, yielding degraded predictions in production [2]. To address these vulnerabilities, Machine Learning Operations (MLOps) establishes systematic practices uniting model development and operations [3]. This research designs, implements, and evaluates an end-to-end MLOps pipeline for residential property valuation in Ames, Iowa.")

    # Section II
    add_heading_1("II. PROBLEM STATEMENT AND OBJECTIVES")
    add_body("The primary analytical objective is to formulate a high-precision regression system capable of estimating the final transaction price (SalePrice) of single-family residences across 80 heterogeneous attributes. Concurrently, the operational objective is to package this predictive system within a fault-tolerant, version-controlled MLOps lifecycle that ensures automated schema verification, traceable experiment logging, and microservice containerization.")

    # Section III
    add_heading_1("III. RELATED WORK AND RESEARCH GAP")
    add_body("Early automated valuation models (AVMs) relied predominantly on Ordinary Least Squares (OLS) and Hedonic Price Modeling [4]. Regularized formulations—notably Ridge (L2) and Lasso (L1) regression—introduced penalty constraints that mitigated multicollinearity and performed automated feature shrinkage [5]. The advent of tree-based ensemble learning shifted the paradigm: Breiman's Random Forests [6] combined bootstrap aggregation with randomized subspace splitting, while gradient boosted decision trees (GBDT) such as XGBoost [8] and LightGBM [9] demonstrated empirical dominance on tabular benchmarks.")
    add_body("However, academic literature frequently divorces machine learning theory from operational sustainability [2]. Existing real estate regression studies predominantly publish static notebook experiments without automated data guards, reproducible pipeline orchestrators, or microservice packaging. This work bridges this gap by delivering a fully integrated, multi-stage MLOps ecosystem applied to tabular housing regression.")

    # Section IV
    add_heading_1("IV. DATASET AND SYSTEM ARCHITECTURE")
    add_body("The project utilizes the benchmark Ames Housing Dataset compiled by Dean De Cock [10], which details residential real estate transactions in Ames, Iowa between 2006 and 2010. The dataset contains 1,460 training records across 81 columns (80 predictors and one continuous target: SalePrice) and 1,459 unseen test instances. The predictors encompass 23 nominal categorical, 23 ordinal categorical, 14 discrete numerical, and 20 continuous numerical variables.")
    add_body("The MLOps architectural workflow is structured into nine decoupled, modular phases: Raw Data Ingestion -> Pandera Schema Guard -> Domain Feature Engineering -> Scikit-Learn Pipeline -> 5-Fold Cross-Validation -> MLflow Tracking & Registry -> Quality Gate Evaluation -> Docker / FastAPI Serving.")

    # Section V
    add_heading_1("V. PREPROCESSING AND SCHEMA GUARD")
    add_body("Preliminary exploratory analysis identified extreme influential outliers. In accordance with De Cock's recommendations [10], two properties with ground living areas exceeding 4,000 square feet selling for unusually low prices (< $300,000) were excised, yielding 1,458 training samples. The target variable, SalePrice, exhibited positive skewness (1.88) and kurtosis. To stabilize variance, a logarithmic transformation was applied: y_transformed = log(1 + y). Predictions are inverted back to currency using exp(y_hat) - 1.")
    add_body("Domain-informed aggregates were engineered: Total Living Area (TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF), Total Bathrooms (FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath), Total Porch Area, House Age, Remodeling Age, and Quality-Area Interaction (QualSF = OverallQual * TotalSF).")
    add_body("To prevent data leakage, transformations were bundled inside a Scikit-Learn ColumnTransformer (RobustScaler for continuous variables and OneHotEncoder for categoricals), generating 279 feature columns. A runtime Pandera schema guard was instituted in src/validate_data.py to enforce non-negative boundaries and strict typing.")

    # Section VI
    add_heading_1("VI. MODEL DEVELOPMENT AND MLFLOW TRACKING")
    add_body("Seven candidate regression algorithms were implemented: Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, LightGBM, and XGBoost. All experiments were orchestrated via src/train_mlflow.py and logged to an SQLite-backed tracking repository (sqlite:///mlflow.db). MLflow systematically captured hyperparameter payloads, cross-validation metrics, serialized artifacts (model.pkl, preprocessor.pkl), and diagnostic plots (residuals, scatterplots, feature importance).")

    # Section VII
    add_heading_1("VII. QUALITY GATES AND REPRODUCIBILITY")
    add_body("Prior to promotion, candidate models must pass an automated quality gate in src/generate_registry_report.py requiring log-scale RMSE <= 0.135 and zero NaNs. Independent back-to-back executions under fixed seed (42) in pipelines/validate_reproducibility.py confirmed 100% deterministic reproducibility (Run 1 RMSE: 0.125821 == Run 2 RMSE: 0.125821). The champion model was formally promoted to Production stage in MLflow as Ames_Housing_Production_Model.")

    # Section VIII
    add_heading_1("VIII. CONTAINERIZED MICROSERVICE SERVING")
    add_body("To serve predictions in real time, a high-performance RESTful API was developed in api/main.py and src/serve_api.py using FastAPI. The service exposes GET /health, POST /predict, and POST /batch_predict endpoints. The serving layer was packaged using an enterprise-ready Dockerfile based on python:3.11-slim, supported by docker-compose.yml and Kubernetes manifests (k8s/deployment.yaml, k8s/service.yaml). In-process testing verified an inference latency below 15ms.")

    # Section IX
    add_heading_1("IX. EXPERIMENTAL RESULTS AND DISCUSSION")
    add_body("Table I summarizes the five-fold cross-validation performance across the seven evaluated regression algorithms on the log-transformed SalePrice.")

    # Table 1
    table_caption_1 = doc.add_paragraph()
    table_caption_1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table_caption_1.paragraph_format.space_before = Pt(8)
    table_caption_1.paragraph_format.space_after = Pt(3)
    c1_run = table_caption_1.add_run("TABLE I. FIVE-FOLD CROSS-VALIDATION PERFORMANCE COMPARISON")
    c1_run.font.name = 'Times New Roman'
    c1_run.font.size = Pt(9)
    c1_run.font.bold = True

    t1 = doc.add_table(rows=8, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_1 = ["Algorithm", "Mean RMSE", "Std. Dev.", "Rank"]
    data_1 = [
        ["XGBoost Regressor", "0.11797", "0.00483", "1"],
        ["Lasso Regression", "0.11870", "0.00915", "2"],
        ["Elastic Net", "0.11870", "0.00916", "3"],
        ["Ridge Regression", "0.11925", "0.00867", "4"],
        ["LightGBM Regressor", "0.12324", "0.00833", "5"],
        ["Gradient Boosting", "0.12333", "0.00692", "6"],
        ["Random Forest", "0.13439", "0.00772", "7"]
    ]
    for col_idx, h in enumerate(headers_1):
        cell = t1.cell(0, col_idx)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            r.font.bold = True

    for row_idx, row_data in enumerate(data_1):
        for col_idx, val in enumerate(row_data):
            cell = t1.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8.5)

    add_body("XGBoost achieved the highest performance, registering the lowest mean RMSE of 0.11797 and the smallest variance (sigma = 0.00483). Table II presents the final evaluation metrics on the test dataset.")

    # Table 2
    table_caption_2 = doc.add_paragraph()
    table_caption_2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table_caption_2.paragraph_format.space_before = Pt(8)
    table_caption_2.paragraph_format.space_after = Pt(3)
    c2_run = table_caption_2.add_run("TABLE II. FINAL PRODUCTION MODEL PERFORMANCE METRICS")
    c2_run.font.name = 'Times New Roman'
    c2_run.font.size = Pt(9)
    c2_run.font.bold = True

    t2 = doc.add_table(rows=4, cols=3)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_2 = ["Metric", "Log-Scale Value", "Currency Equiv. ($)"]
    data_2 = [
        ["Root Mean Squared Error", "0.11797", "$13,617.59"],
        ["Mean Absolute Error (MAE)", "0.08081", "$9,412.30"],
        ["Coefficient of Det. (R2)", "0.91215", "0.91215"]
    ]
    for col_idx, h in enumerate(headers_2):
        cell = t2.cell(0, col_idx)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            r.font.bold = True

    for row_idx, row_data in enumerate(data_2):
        for col_idx, val in enumerate(row_data):
            cell = t2.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8.5)

    add_body("Feature attribution identified QualSF, OverallQual, GrLivArea, TotalBsmtSF, and Neighborhood as the dominant drivers of residential valuation.")

    # Section X
    add_heading_1("X. LIMITATIONS AND FUTURE WORK")
    add_body("The Ames dataset reflects localized Midwestern demographics; geographical transferability requires retraining on regional datasets. Future work will incorporate continuous drift monitoring with Evidently AI and explainable AI dashboards integrating SHAP values into the FastAPI interface.")

    # Section XI
    add_heading_1("XI. CONCLUSION")
    add_body("This study demonstrated an end-to-end MLOps pipeline for house price prediction uniting domain feature engineering, Pandera schema validation, MLflow lifecycle tracking, and containerized FastAPI/Docker serving. XGBoost achieved a cross-validated RMSE of 0.11797 and R2 of 0.91215 with 100% deterministic reproducibility, providing a validated blueprint for enterprise tabular machine learning systems.")

    # References
    add_heading_1("REFERENCES")
    refs = [
        "[1] D. De Cock, 'Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project,' J. Stat. Educ., vol. 19, no. 3, 2011.",
        "[2] D. Sculley et al., 'Hidden technical debt in machine learning systems,' in Proc. NeurIPS, 2015, pp. 2503-2511.",
        "[3] S. Alla and S. K. Adari, Beginning MLOps with Python. Berkeley, CA: Apress, 2021.",
        "[4] S. C. Bourassa et al., 'Predicting house prices with spatial hedonic models,' Real Estate Econ., vol. 38, no. 4, pp. 723-748, 2010.",
        "[5] R. Tibshirani, 'Regression shrinkage and selection via the lasso,' J. R. Stat. Soc. B, vol. 58, no. 1, pp. 267-288, 1996.",
        "[6] L. Breiman, 'Random forests,' Mach. Learn., vol. 45, no. 1, pp. 5-32, 2001.",
        "[7] J. H. Friedman, 'Greedy function approximation: A gradient boosting machine,' Ann. Stat., vol. 29, no. 5, pp. 1189-1232, 2001.",
        "[8] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in Proc. ACM SIGKDD, 2016, pp. 785-794.",
        "[9] G. Ke et al., 'LightGBM: A highly efficient gradient boosting decision tree,' in Proc. NeurIPS, vol. 30, 2017, pp. 3146-3154.",
        "[10] Kaggle, 'House Prices: Advanced Regression Techniques,' 2016. [Online]. Available: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.",
        "[11] M. Zaharia et al., 'Accelerating the machine learning lifecycle with MLflow,' IEEE Data Eng. Bull., vol. 41, no. 4, pp. 39-45, 2018.",
        "[12] S. Ramirez, 'FastAPI: Modern, fast web framework for building APIs with Python,' 2020. [Online]. Available: https://fastapi.tiangolo.com."
    ]
    for r_text in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.first_line_indent = Inches(-0.2)
        run = p.add_run(r_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)

    # About the Author (Strictly Text Only - No Photo)
    add_heading_1("ABOUT THE AUTHOR")
    author_bio = doc.add_paragraph()
    author_bio.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    author_bio.paragraph_format.space_before = Pt(4)
    author_bio.paragraph_format.space_after = Pt(4)
    
    b_name = author_bio.add_run("Nidamanuru Shanmukha Ram ")
    b_name.font.name = 'Times New Roman'
    b_name.font.size = Pt(9)
    b_name.font.bold = True
    
    b_text = author_bio.add_run("is an Artificial Intelligence and Machine Learning scholar in the Department of Artificial Intelligence and Machine Learning at Vignan's Foundation for Science, Technology and Research, Bapatla, Andhra Pradesh, India. His research interests encompass enterprise Machine Learning Operations (MLOps), predictive tabular regression, automated model governance, and scalable cloud containerization architectures.")
    b_text.font.name = 'Times New Roman'
    b_text.font.size = Pt(9)

    # Save to both locations
    desktop_path = r"C:\Users\shanm\OneDrive\Desktop\NEW_HOUSE_PRICE_RESEARCH_PAPER.docx"
    reports_path = r"C:\Users\shanm\OneDrive\Desktop\housepricepredict\reports\NEW_RESEARCH_PAPER_IEEE.docx"
    
    doc.save(desktop_path)
    doc.save(reports_path)
    print(f"Successfully generated DOCX files at:\n1. {desktop_path}\n2. {reports_path}")

if __name__ == "__main__":
    create_ieee_paper()
