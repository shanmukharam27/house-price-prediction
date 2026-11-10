"""
Verification script for Lab 10 & 11: Ames House Price REST API & Container
Tests both in-process inference and live HTTP requests.
"""

import sys
import numpy as np
import pandas as pd
import requests

SAMPLE_HOUSE = {
    'MSSubClass': 60, 'MSZoning': 'RL', 'LotFrontage': 65, 'LotArea': 8450, 
    'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl', 
    'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 
    'Neighborhood': 'CollgCr', 'Condition1': 'Norm', 'Condition2': 'Norm', 
    'BldgType': '1Fam', 'HouseStyle': '2Story', 'OverallQual': 7, 'OverallCond': 5, 
    'YearBuilt': 2003, 'YearRemodAdd': 2003, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 
    'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'BrkFace', 
    'MasVnrArea': 196, 'ExterQual': 'Gd', 'ExterCond': 'TA', 'Foundation': 'PConc', 
    'BsmtQual': 'Gd', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ', 
    'BsmtFinSF1': 706, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 150, 
    'TotalBsmtSF': 856, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y', 
    'Electrical': 'SBrkr', '1stFlrSF': 856, '2ndFlrSF': 854, 'LowQualFinSF': 0, 
    'GrLivArea': 1710, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 2, 
    'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'Gd', 
    'TotRmsAbvGrd': 8, 'Functional': 'Typ', 'Fireplaces': 0, 'FireplaceQu': 'None', 
    'GarageType': 'Attchd', 'GarageYrBlt': 2003, 'GarageFinish': 'RFn', 
    'GarageCars': 2, 'GarageArea': 548, 'GarageQual': 'TA', 'GarageCond': 'TA', 
    'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 61, 'EnclosedPorch': 0, 
    '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'None', 
    'Fence': 'None', 'MiscFeature': 'None', 'MiscVal': 0, 'MoSold': 2, 
    'YrSold': 2008, 'SaleType': 'WD', 'SaleCondition': 'Normal'
}

def test_in_process():
    print("\n[TEST 1] Testing In-Process House Price Model & Preprocessing Pipeline...")
    try:
        import joblib
        from src.preprocess_pipeline import engineer_features

        preprocessor = joblib.load("models/preprocessor.pkl")
        model = joblib.load("models/ridge_model.pkl")

        df = pd.DataFrame([SAMPLE_HOUSE])
        engineered = engineer_features(df)
        X_trans = preprocessor.transform(engineered)

        log_pred = float(model.predict(X_trans)[0])
        pred_price = float(np.expm1(log_pred))

        print(f"  -> Predicted Log SalePrice : {log_pred:.4f}")
        print(f"  -> Predicted Sale Price ($): ${pred_price:,.2f}")
        print("  -> Status: PASSED\n")
        return True
    except Exception as exc:
        print(f"  -> Status: FAILED ({exc})\n")
        return False

def test_http_endpoint(base_url="http://127.0.0.1:8000"):
    print(f"\n[TEST 2] Testing Live HTTP Endpoints at {base_url}...")
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        if health_resp.status_code != 200:
            print(f"  -> Health check failed: {health_resp.status_code}")
            return False
        print(f"  -> Health Check: {health_resp.json()}")

        pred_resp = requests.post(f"{base_url}/predict", json=SAMPLE_HOUSE, timeout=5)
        if pred_resp.status_code != 200:
            print(f"  -> Predict failed: {pred_resp.status_code}")
            return False
        print(f"  -> Predict Response: {pred_resp.json()}")
        print("  -> Status: PASSED\n")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  -> Could not connect to {base_url}. (Ensure the server is running).")
        return False
    except Exception as exc:
        print(f"  -> Status: FAILED ({exc})\n")
        return False

if __name__ == "__main__":
    passed_local = test_in_process()
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        passed_http = test_http_endpoint()
        sys.exit(0 if (passed_local and passed_http) else 1)
    sys.exit(0 if passed_local else 1)
