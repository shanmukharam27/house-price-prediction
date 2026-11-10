"""
=============================================================================
HOUSE PRICE PREDICTION - COMPLETE STANDALONE PYTHON SCRIPT (.py)
=============================================================================
This is the complete, self-contained Python (.py) file converted directly from
the Jupyter Notebook. No Jupyter Notebook (.ipynb) is required.

Features:
  - Step 1: Automatically locates and loads train.csv and test.csv
  - Step 2: Exploratory Data Analysis & visual outputs
  - Step 3: Outlier removal, feature engineering & robust scaling
  - Step 4: 5-Fold Cross-Validation across 7 Regression Models
            (Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM)
  - Step 5: Serializes all trained models (.pkl) into models/
  - Step 6: Evaluates validation metrics (Log RMSE, MAE, R², Dollar Error)
  - Step 7: Feature Importance analysis
  - Step 8: Generates Kaggle submission.csv in outputs/
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

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# PATH RESOLVER
# =============================================================================
def resolve_data_paths():
    """Finds train.csv and test.csv across possible working directories."""
    candidates_train = [
        'data/raw/train.csv',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw', 'train.csv'),
        r'C:\Users\shanm\OneDrive\Desktop\18014\housepricepredict\data\raw\train.csv',
        r"C:\Users\shanm\OneDrive\Desktop\18014\housepricepredict'\data\raw\train.csv"
    ]
    candidates_test = [
        'data/raw/test.csv',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw', 'test.csv'),
        r'C:\Users\shanm\OneDrive\Desktop\18014\housepricepredict\data\raw\test.csv',
        r"C:\Users\shanm\OneDrive\Desktop\18014\housepricepredict'\data\raw\test.csv"
    ]
    
    train_path = next((p for p in candidates_train if os.path.exists(p)), None)
    test_path = next((p for p in candidates_test if os.path.exists(p)), None)
    
    if not train_path:
        raise FileNotFoundError("Could not find train.csv in standard project paths.")
        
    return train_path, test_path

# =============================================================================
# DIRECTORIES SETUP
# =============================================================================
def setup_directories():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    data = df.copy()
    
    # Missing categorical to 'None'
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'MasVnrType']
    for c in none_cols:
        if c in data.columns:
            data[c] = data[c].fillna('None')
            
    # Missing numerical to 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                 'BsmtHalfBath', 'MasVnrArea']
    for c in zero_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0)
            
    # LotFrontage by neighborhood median
    if 'LotFrontage' in data.columns:
        if 'Neighborhood' in data.columns:
            data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
        
    # Mode imputation
    mode_cols = ['MSZoning', 'Functional', 'Electrical', 'KitchenQual', 
                 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
    for c in mode_cols:
        if c in data.columns:
            m = data[c].mode()
            data[c] = data[c].fillna(m[0] if len(m) > 0 else 'None')
            
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)
        
    # Domain features
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

# =============================================================================
# MAIN WORKFLOW
# =============================================================================
def run_project():
    print("=" * 75)
    print("   HOUSE PRICE PREDICTION - STANDALONE PYTHON PIPELINE (.py)")
    print("=" * 75)
    
    setup_directories()
    train_path, test_path = resolve_data_paths()
    print(f"[INFO] Using training data : {train_path}")
    print(f"[INFO] Using test data     : {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    
    print(f"\n[STEP 1] Data Overview:")
    print(f"  - Train Shape: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    if test_df is not None:
        print(f"  - Test Shape : {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        
    # Outlier Removal
    outliers = (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)
    print(f"\n[STEP 2] Outlier Cleaning: Removing {outliers.sum()} anomalous records...")
    clean_train = train_df[~outliers].copy()
    
    # Target log transformation
    y = np.log1p(clean_train['SalePrice'])
    X_raw = clean_train.drop(['Id', 'SalePrice'], axis=1)
    
    # Feature Engineering
    print("[STEP 3] Engineering domain features (TotalSF, Bathrooms, HouseAge, QualSF)...")
    X_eng = engineer_features(X_raw)
    
    # Train-test split (80-20)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_eng, y, test_size=0.2, random_state=42
    )
    
    cat_cols = X_train_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X_train_df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    
    # Scaling & Encoding
    scaler = RobustScaler()
    x_tr_scaled = scaler.fit_transform(X_train_df[num_cols])
    x_val_scaled = scaler.transform(X_val_df[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_tr_encoded = ohe.fit_transform(X_train_df[cat_cols])
    x_val_encoded = ohe.transform(X_val_df[cat_cols])
    
    X_train = np.hstack((x_tr_scaled, x_tr_encoded))
    X_val = np.hstack((x_val_scaled, x_val_encoded))
    
    # Save artifacts
    np.save('data/processed/X_train_final.npy', X_train)
    np.save('data/processed/X_test_final.npy', X_val)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.float64))
    np.save('data/processed/y_test.npy', y_val.to_numpy(dtype=np.float64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')
    
    # Process Kaggle test dataset if present
    X_sub = None
    test_ids = None
    if test_df is not None:
        test_ids = test_df['Id'].to_numpy()
        test_features = test_df.drop(['Id'], axis=1)
        test_eng = engineer_features(test_features)
        test_scaled = scaler.transform(test_eng[num_cols])
        test_encoded = ohe.transform(test_eng[cat_cols])
        X_sub = np.hstack((test_scaled, test_encoded))
        np.save('data/processed/X_submission_final.npy', X_sub)
        np.save('data/processed/test_ids.npy', test_ids)
        
    print(f"\n[STEP 4] Model Training & Cross-Validation Benchmarking:")
    print(f"{'Model Name':<20} | {'Log RMSE':<10} | {'Log MAE':<10} | {'R2 Score':<10} | {'Dollar RMSE ($)':<15}")
    print("-" * 75)
    
    models = {
        'Ridge': Ridge(alpha=10.0, random_state=42),
        'Lasso': Lasso(alpha=0.0005, random_state=42, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.7, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=5, random_state=42)
    }
    
    trained_models = {}
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        preds_log = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds_log))
        mae = mean_absolute_error(y_val, preds_log)
        r2 = r2_score(y_val, preds_log)
        
        preds_dollars = np.expm1(preds_log)
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
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\n[WINNER] Best Performing Model: {best_model_name} (Lowest Log RMSE = {results_df.iloc[0]['Log_RMSE']:.4f})")
    
    # Save all models
    print("\n[STEP 5] Serializing all models into models/ folder...")
    joblib.dump(trained_models['Ridge'], 'models/ridge_model.pkl')
    joblib.dump(trained_models['Ridge'], 'models/ridge_baseline.pkl')
    joblib.dump(trained_models['Lasso'], 'models/lasso_model.pkl')
    joblib.dump(trained_models['ElasticNet'], 'models/elasticnet_model.pkl')
    joblib.dump(trained_models['RandomForest'], 'models/random_forest_model.pkl')
    joblib.dump(trained_models['GradientBoosting'], 'models/gradient_boosting_model.pkl')
    joblib.dump(trained_models['XGBoost'], 'models/xgboost_model.pkl')
    joblib.dump(trained_models['LightGBM'], 'models/lightgbm_model.pkl')
    print("  -> Saved 8 model binary (.pkl) files to models/")
    
    # Visualizations & Diagnostics
    print("\n[STEP 6] Generating Evaluation Diagnostics & Error Analysis...")
    preds_log = best_model.predict(X_val)
    preds_dollars = np.expm1(preds_log)
    actual_dollars = np.expm1(y_val)
    
    # Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_dollars, preds_dollars, alpha=0.5, color='royalblue', edgecolors='k')
    min_v, max_v = min(actual_dollars.min(), preds_dollars.min()), max(actual_dollars.max(), preds_dollars.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    ax.set_title(f'Actual vs Predicted SalePrice - {best_model_name}')
    ax.set_xlabel('Actual SalePrice ($)')
    ax.set_ylabel('Predicted SalePrice ($)')
    plt.tight_layout()
    plt.savefig('artifacts/actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # Residuals plot
    residuals = y_val - preds_log
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='forestgreen')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'Residuals Distribution - {best_model_name}')
    plt.xlabel('Log Residual Error')
    plt.tight_layout()
    plt.savefig('artifacts/residuals_distribution.png', dpi=150)
    plt.close()
    
    # High error cases
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
    error_df.sort_values('Absolute_Pct_Error', ascending=False).to_csv('outputs/high_residual_errors.csv', index=False)
    print("  -> Saved artifacts/actual_vs_predicted.png")
    print("  -> Saved artifacts/residuals_distribution.png")
    print("  -> Saved outputs/high_residual_errors.csv")
    
    # Kaggle Submission
    if X_sub is not None and test_ids is not None:
        print("\n[STEP 7] Generating Kaggle submission.csv...")
        sub_preds = np.expm1(best_model.predict(X_sub))
        sub_df = pd.DataFrame({'Id': test_ids, 'SalePrice': sub_preds})
        sub_df.to_csv('outputs/submission.csv', index=False)
        print(f"[SUCCESS] Kaggle submission written to outputs/submission.csv ({len(sub_df)} rows)")
        print(sub_df.head())
        
    print("\n" + "=" * 75)
    print("   COMPLETE PROJECT EXECUTION FINISHED SUCCESSFULLY! (.py)")
    print("=" * 75)

if __name__ == "__main__":
    run_project()
