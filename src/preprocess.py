import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder

def engineer_features(df):
    """Creates domain-specific engineered features for the Ames Housing dataset."""
    data = df.copy()
    
    # Missing values categorical None
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'MasVnrType']
    for c in none_cols:
        if c in data.columns:
            data[c] = data[c].fillna('None')
            
    # Numerical 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                 'BsmtHalfBath', 'MasVnrArea']
    for c in zero_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0)
            
    if 'LotFrontage' in data.columns:
        if 'Neighborhood' in data.columns:
            data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
        
    mode_cols = ['MSZoning', 'Functional', 'Electrical', 'KitchenQual', 
                 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
    for c in mode_cols:
        if c in data.columns:
            m = data[c].mode()
            data[c] = data[c].fillna(m[0] if len(m) > 0 else 'None')
            
    if 'MSSubClass' in data.columns:
        data['MSSubClass'] = data['MSSubClass'].astype(str)
        
    # Domain feature engineering
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

def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    
    # 1. Load the raw data
    data_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Outlier Removal (standard Kaggle Ames benchmark)
    outliers = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
    print(f"Removing {outliers.sum()} extreme outlier records...")
    df = df[~outliers].copy()
    
    # 3. Target Variable Extraction & Transformation (Log1p for normal distribution)
    y = np.log1p(df['SalePrice'])
    X = df.drop(['Id', 'SalePrice'], axis=1)
    
    # 4. Feature Engineering
    X = engineer_features(X)
    
    # 5. Train-Test Split (Done before fitting transformations to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 6. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    
    # 7. Scale & Encode
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    
    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    
    # 8. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.float64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.float64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')
    
    # Preprocess test.csv for Kaggle submission if available
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_ids = test_df['Id'].to_numpy()
        test_X = test_df.drop(['Id'], axis=1)
        test_X = engineer_features(test_X)
        test_scaled = scaler.transform(test_X[num_cols])
        test_encoded = ohe.transform(test_X[cat_cols])
        X_sub_final = np.hstack((test_scaled, test_encoded))
        np.save('data/processed/X_submission_final.npy', X_sub_final)
        np.save('data/processed/test_ids.npy', test_ids)
        print(f"Kaggle test set processed successfully (Shape: {X_sub_final.shape})")
        
    # Save Metadata
    metadata = {
        "dataset_name": "Ames Iowa House Price Prediction",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "total_features": int(X_train_final.shape[1]),
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_variable": "SalePrice",
        "target_transformation": "log1p"
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()
