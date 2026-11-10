import pytest
from starlette.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

VALID_HOUSE = {
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

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Ames House Price Prediction API"

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True

def test_predict_endpoint(client):
    response = client.post("/predict", json=VALID_HOUSE)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_usd" in data
    assert data["predicted_price_usd"] > 50000
    assert data["status"] == "success"
