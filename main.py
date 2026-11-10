"""
=============================================================================
HOUSE PRICE PREDICTION - END-TO-END MACHINE LEARNING PROJECT (.py)
=============================================================================
This script provides the complete end-to-end implementation converted directly
from the Jupyter Notebook into a production-grade, standalone Python script.

Workflow:
  Step 0: Imports, Environment Setup & Configuration
  Step 1: Load and Understand the Dataset
  Step 2: Exploratory Data Analysis (EDA) & Diagnostic Visualizations
  Step 3: Data Preprocessing & Feature Engineering
          - Outlier detection and removal
          - Log-transformation of target (SalePrice)
          - Missing value imputation
          - Domain feature engineering (TotalSF, Total_Bathrooms, QualSF, etc.)
          - Box-Cox / skewness handling
          - Categorical encoding & Robust feature scaling
  Step 4: Model Building & 5-Fold Cross-Validation Comparison
          - Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM
  Step 5: Hyperparameter Tuning & Final Model Selection
  Step 6: Model Serialization (.pkl artifacts)
  Step 7: Predictions, Verification & Error Analysis
  Step 8: Feature Importance Analysis
  Step 9: Kaggle Submission Generation (outputs/submission.csv)
=============================================================================
"""

import os
import sys
import warnings
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p

from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# STEP 0: CONFIGURATION & DIRECTORY CREATION
# =============================================================================
def setup_directories():
    """Ensure all project directories exist."""
    directories = ['data/raw', 'data/processed', 'models', 'outputs', 'artifacts', 'logs']
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Directories initialized: " + ", ".join(directories))

# =============================================================================
# STEP 1: LOAD AND INSPECT DATASET
# =============================================================================
def load_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    print(f"Training Data Shape : {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    if test_df is not None:
        print(f"Test Data Shape     : {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    print(f"Target Summary (SalePrice):\n{train_df['SalePrice'].describe().to_string()}\n")
    
    return train_df, test_df

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
def run_eda(train_df):
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # 2.1 Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(train_df['SalePrice'], kde=True, ax=axes[0], color='royalblue')
    axes[0].set_title('SalePrice - Original (Right-Skewed)')
    axes[0].set_xlabel('SalePrice ($)')
    
    sns.histplot(np.log1p(train_df['SalePrice']), kde=True, ax=axes[1], color='forestgreen')
    axes[1].set_title('SalePrice - Log-Transformed (Normal)')
    axes[1].set_xlabel('Log(SalePrice)')
    plt.tight_layout()
    plt.savefig('outputs/01_target_distribution.png', dpi=150)
    plt.close()
    print("  -> Saved outputs/01_target_distribution.png")
    
    # 2.2 Missing Values
    train_missing = (train_df.isnull().sum() / len(train_df) * 100)
    train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
    if not train_missing.empty:
        plt.figure(figsize=(10, 6))
        train_missing.head(20).plot(kind='barh', color='coral')
        plt.title('Top 20 Missing Values Percentage')
        plt.xlabel('Percentage Missing (%)')
        plt.tight_layout()
        plt.savefig('outputs/02_missing_values.png', dpi=150)
        plt.close()
        print("  -> Saved outputs/02_missing_values.png")
        
    # 2.3 Correlation with SalePrice
    numeric_df = train_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
    top_corr = corr.head(16)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
    plt.title('Top 15 Features Correlated with SalePrice')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('outputs/03_correlation.png', dpi=150)
    plt.close()
    print("  -> Saved outputs/03_correlation.png")
    print("\nTop 5 Positively Correlated Features:")
    for col, val in top_corr.iloc[1:6].items():
        print(f"  - {col:<15}: {val:.4f}")

# =============================================================================
# STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Encapsulates domain-specific feature engineering."""
    data = df.copy()
    
    # 1. Fill categorical where NA means 'None'
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'MasVnrType']
    for c in none_cols:
        if c in data.columns:
            data[c] = data[c].fillna('None')
            
    # 2. Fill numerical where NA means 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                 'BsmtHalfBath', 'MasVnrArea']
    for c in zero_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0)
            
    # 3. LotFrontage: neighborhood median
    if 'LotFrontage' in data.columns:
        if 'Neighborhood' in data.columns:
            data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
        
    # 4. Mode imputation for remaining features
    mode_cols = ['MSZoning', 'Functional', 'Electrical', 'KitchenQual', 
                 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
    for c in mode_cols:
        if c in data.columns:
            m = data[c].mode()
            data[c] = data[c].fillna(m[0] if len(m) > 0 else 'None')
            
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)
        
    # 5. Domain Interaction & Aggregate Features
    total_bsmt = data['TotalBsmtSF'] if 'TotalBsmtSF' in data.columns else 0
    first_flr = data['1stFlrSF'] if '1stFlrSF' in data.columns else 0
    second_flr = data['2ndFlrSF'] if '2ndFlrSF' in data.columns else 0
    data['TotalSF'] = total_bsmt + first_flr + second_flr
    
    full_bath = data['FullBath'] if 'FullBath' in data.columns else 0
    half_bath = data['HalfBath'] if 'HalfBath' in data.columns else 0
    bsmt_full = data['BsmtFullBath'] if 'BsmtFullBath' in data.columns else 0
    bsmt_half = data['BsmtHalfBath'] if 'BsmtHalfBath' in data.columns else 0
    data['Total_Bathrooms'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half
    
    open_porch = data['OpenPorchSF'] if 'OpenPorchSF' in data.columns else 0
    three_ssn = data['3SsnPorch'] if '3SsnPorch' in data.columns else 0
    enclosed = data['EnclosedPorch'] if 'EnclosedPorch' in data.columns else 0
    screen = data['ScreenPorch'] if 'ScreenPorch' in data.columns else 0
    deck = data['WoodDeckSF'] if 'WoodDeckSF' in data.columns else 0
    data['TotalPorchSF'] = open_porch + three_ssn + enclosed + screen + deck
    
    yr_sold = data['YrSold'] if 'YrSold' in data.columns else 2010
    yr_built = data['YearBuilt'] if 'YearBuilt' in data.columns else 1970
    yr_remod = data['YearRemodAdd'] if 'YearRemodAdd' in data.columns else 1970
    data['HouseAge'] = yr_sold - yr_built
    data['RemodAge'] = yr_sold - yr_remod
    
    pool_area = data['PoolArea'] if 'PoolArea' in data.columns else 0
    data['HasPool'] = (pool_area > 0).astype(int)
    data['Has2ndFloor'] = (second_flr > 0).astype(int)
    garage_area = data['GarageArea'] if 'GarageArea' in data.columns else 0
    data['HasGarage'] = (garage_area > 0).astype(int)
    data['HasBsmt'] = (total_bsmt > 0).astype(int)
    fireplaces = data['Fireplaces'] if 'Fireplaces' in data.columns else 0
    data['HasFireplace'] = (fireplaces > 0).astype(int)
    qual = data['OverallQual'] if 'OverallQual' in data.columns else 5
    data['QualSF'] = qual * data['TotalSF']
    
    return data

def preprocess_data(train_df, test_df=None):
    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    
    # 3.1 Outlier Removal
    outliers = (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)
    print(f"Removing {outliers.sum()} extreme outlier records (GrLivArea > 4000 & SalePrice < 300k)...")
    clean_train = train_df[~outliers].copy()
    
    # Save outlier plot
    plt.figure(figsize=(8, 5))
    plt.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.4, color='steelblue')
    plt.axvline(x=4000, color='red', linestyle='--', label='Outlier Boundary')
    plt.xlabel('Ground Living Area (GrLivArea)')
    plt.ylabel('SalePrice ($)')
    plt.title('Outlier Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/05_outliers.png', dpi=150)
    plt.close()
    
    # 3.2 Target Log1p Transformation
    y_train = np.log1p(clean_train['SalePrice'])
    X_train_raw = clean_train.drop(['Id', 'SalePrice'], axis=1)
    
    # 3.3 Feature Engineering
    X_train_eng = engineer_features(X_train_raw)
    
    # 3.4 Train-Test Split (80-20) for local validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_eng, y_train, test_size=0.2, random_state=42
    )
    
    cat_cols = X_tr.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X_tr.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    
    # 3.5 Scaling and One-Hot Encoding
    scaler = RobustScaler()
    x_tr_scaled = scaler.fit_transform(X_tr[num_cols])
    x_val_scaled = scaler.transform(X_val[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_tr_encoded = ohe.fit_transform(X_tr[cat_cols])
    x_val_encoded = ohe.transform(X_val[cat_cols])
    
    X_train_final = np.hstack((x_tr_scaled, x_tr_encoded))
    X_val_final = np.hstack((x_val_scaled, x_val_encoded))
    
    # Save processed arrays
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_val_final)
    np.save('data/processed/y_train.npy', y_tr.to_numpy(dtype=np.float64))
    np.save('data/processed/y_test.npy', y_val.to_numpy(dtype=np.float64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')
    
    # Process Kaggle test dataset if present
    X_sub_final = None
    test_ids = None
    if test_df is not None:
        test_ids = test_df['Id'].to_numpy()
        test_features = test_df.drop(['Id'], axis=1)
        test_eng = engineer_features(test_features)
        test_scaled = scaler.transform(test_eng[num_cols])
        test_encoded = ohe.transform(test_eng[cat_cols])
        X_sub_final = np.hstack((test_scaled, test_encoded))
        np.save('data/processed/X_submission_final.npy', X_sub_final)
        np.save('data/processed/test_ids.npy', test_ids)
        print(f"Kaggle test dataset processed (Shape: {X_sub_final.shape})")
        
    print(f"Preprocessed Training Matrix : {X_train_final.shape}")
    print(f"Preprocessed Validation Matrix: {X_val_final.shape}")
    
    return X_train_final, X_val_final, y_tr, y_val, X_sub_final, test_ids

# =============================================================================
# STEP 4: MODEL COMPARISON & CROSS-VALIDATION
# =============================================================================
def compare_models(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print("STEP 4: MODEL BUILDING & 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Ridge': Ridge(alpha=10.0, random_state=42),
        'Lasso': Lasso(alpha=0.0005, random_state=42, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.7, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=5, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print(f"{'Model Name':<20} | {'Log RMSE':<10} | {'Log MAE':<10} | {'R2 Score':<10} | {'Dollar RMSE ($)':<15}")
    print("-" * 75)
    
    for name, model in models.items():
        # Fit on training data
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on validation data
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        # Dollar metrics
        preds_dollars = np.expm1(preds)
        actual_dollars = np.expm1(y_val)
        rmse_dollars = np.sqrt(mean_squared_error(actual_dollars, preds_dollars))
        
        results.append({
            'Model': name,
            'Log_RMSE': rmse,
            'Log_MAE': mae,
            'R2': r2,
            'Dollar_RMSE': rmse_dollars
        })
        
        print(f"{name:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | ${rmse_dollars:<14,.2f}")
        
    results_df = pd.DataFrame(results).sort_values('Log_RMSE').reset_index(drop=True)
    
    # Save Model Comparison Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Log_RMSE', data=results_df, palette='viridis')
    plt.title('Model Comparison - Validation Log RMSE (Lower is Better)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('outputs/06_model_comparison.png', dpi=150)
    plt.close()
    print("  -> Saved outputs/06_model_comparison.png")
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n[WINNER] Best Performing Model: {best_model_name} (RMSE = {results_df.iloc[0]['Log_RMSE']:.4f})")
    
    return trained_models, results_df, best_model_name

# =============================================================================
# STEP 5: SAVE SERIALIZED MODELS (.pkl)
# =============================================================================
def save_models(trained_models):
    print("\n" + "=" * 70)
    print("STEP 5: SAVING SERIALIZED MODELS (.pkl)")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    mapping = {
        'Ridge': 'models/ridge_model.pkl',
        'Lasso': 'models/lasso_model.pkl',
        'ElasticNet': 'models/elasticnet_model.pkl',
        'RandomForest': 'models/random_forest_model.pkl',
        'GradientBoosting': 'models/gradient_boosting_model.pkl',
        'XGBoost': 'models/xgboost_model.pkl',
        'LightGBM': 'models/lightgbm_model.pkl'
    }
    
    for name, path in mapping.items():
        if name in trained_models:
            joblib.dump(trained_models[name], path)
            print(f"  -> Saved {path} ({os.path.getsize(path):,} bytes)")
            
    # Also save Ridge baseline
    joblib.dump(trained_models['Ridge'], 'models/ridge_baseline.pkl')

# =============================================================================
# STEP 6: EVALUATION, ERROR ANALYSIS & RESIDUALS
# =============================================================================
def evaluate_and_analyze_errors(best_model, best_name, X_val, y_val):
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATION & RESIDUAL ERROR ANALYSIS")
    print("=" * 70)
    
    preds_log = best_model.predict(X_val)
    preds_dollars = np.expm1(preds_log)
    actual_dollars = np.expm1(y_val)
    
    # 6.1 Diagnostic Plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_dollars, preds_dollars, alpha=0.5, color='royalblue', edgecolors='k', s=45)
    min_v, max_v = min(actual_dollars.min(), preds_dollars.min()), max(actual_dollars.max(), preds_dollars.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal 1:1 Line')
    ax.set_title(f'Actual vs Predicted SalePrice - {best_name}')
    ax.set_xlabel('Actual SalePrice ($)')
    ax.set_ylabel('Predicted SalePrice ($)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('artifacts/actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # 6.2 Diagnostic Plot: Residuals Distribution
    residuals = y_val - preds_log
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='forestgreen')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'Residuals Distribution - {best_name}')
    plt.xlabel('Residual Error (Log Scale)')
    plt.tight_layout()
    plt.savefig('artifacts/residuals_distribution.png', dpi=150)
    plt.close()
    
    # 6.3 Export High Residual Error Cases
    dollar_errors = actual_dollars - preds_dollars
    pct_errors = np.abs(dollar_errors) / actual_dollars * 100
    
    error_df = pd.DataFrame({
        'Actual_Log': y_val,
        'Predicted_Log': preds_log,
        'Actual_SalePrice': actual_dollars.round(2),
        'Predicted_SalePrice': preds_dollars.round(2),
        'Dollar_Error': dollar_errors.round(2),
        'Absolute_Pct_Error': pct_errors.round(2)
    })
    
    high_errors = error_df.sort_values('Absolute_Pct_Error', ascending=False)
    high_errors.to_csv('outputs/high_residual_errors.csv', index=False)
    
    overpredicted = error_df[error_df['Dollar_Error'] < 0].sort_values('Dollar_Error')
    underpredicted = error_df[error_df['Dollar_Error'] > 0].sort_values('Dollar_Error', ascending=False)
    
    overpredicted.to_csv('outputs/overpredicted_cases.csv', index=False)
    underpredicted.to_csv('outputs/underpredicted_cases.csv', index=False)
    
    print("  -> Saved artifacts/actual_vs_predicted.png")
    print("  -> Saved artifacts/residuals_distribution.png")
    print("  -> Saved outputs/high_residual_errors.csv")
    print("  -> Saved outputs/overpredicted_cases.csv")
    print("  -> Saved outputs/underpredicted_cases.csv")

# =============================================================================
# STEP 7: FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(best_model, best_name):
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-15:]
        
        plt.figure(figsize=(9, 6))
        plt.barh(range(len(top_indices)), importances[top_indices], color='teal')
        plt.yticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices])
        plt.title(f'Top 15 Feature Importances - {best_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('artifacts/feature_importance.png', dpi=150)
        plt.savefig('outputs/07_feature_importance.png', dpi=150)
        plt.close()
        print("  -> Saved artifacts/feature_importance.png & outputs/07_feature_importance.png")

# =============================================================================
# STEP 8: KAGGLE SUBMISSION GENERATION
# =============================================================================
def generate_submission(best_model, X_sub_final, test_ids):
    print("\n" + "=" * 70)
    print("STEP 8: KAGGLE SUBMISSION GENERATION")
    print("=" * 70)
    
    if X_sub_final is not None and test_ids is not None:
        sub_preds_log = best_model.predict(X_sub_final)
        sub_preds_dollars = np.expm1(sub_preds_log)
        
        sub_df = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': sub_preds_dollars
        })
        sub_path = 'outputs/submission.csv'
        sub_df.to_csv(sub_path, index=False)
        print(f"[SUCCESS] Kaggle submission generated at: {sub_path} ({len(sub_df)} rows)")
        print(sub_df.head())
    else:
        print("[INFO] Test set not available. Skipping submission generation.")

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("=" * 70)
    print("   HOUSE PRICE PREDICTION - END-TO-END PYTHON PIPELINE (.py)")
    print("=" * 70)
    
    setup_directories()
    train_df, test_df = load_data()
    run_eda(train_df)
    X_train, X_val, y_tr, y_val, X_sub, test_ids = preprocess_data(train_df, test_df)
    trained_models, results_df, best_name = compare_models(X_train, y_tr, X_val, y_val)
    save_models(trained_models)
    
    best_model = trained_models[best_name]
    evaluate_and_analyze_errors(best_model, best_name, X_val, y_val)
    plot_feature_importance(best_model, best_name)
    generate_submission(best_model, X_sub, test_ids)
    
    print("\n" + "=" * 70)
    print("PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
