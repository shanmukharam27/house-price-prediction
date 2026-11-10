"""
Lab 10 & 11: Production REST API for Ames House Price Prediction
Serves real-time regression predictions via FastAPI, designed for Docker containerization.
"""

import os
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess_pipeline import engineer_features

app = FastAPI(
    title="Ames House Price Prediction Service",
    description="Containerized REST API for real-time house price estimation (MLOps Lab 10 & 11)",
    version="1.0.0",
)

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# Global model and preprocessor references
model = None
preprocessor = None
config = {}


def load_artifacts():
    global model, preprocessor, config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "paths": {
                "model": "models/ridge_model.pkl",
                "preprocessor": "models/preprocessor.pkl",
            }
        }

    model_path = os.getenv("MODEL_PATH", config.get("paths", {}).get("model", "models/ridge_model.pkl"))
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", config.get("paths", {}).get("preprocessor", "models/preprocessor.pkl"))

    # Fallback to deployment folder if present
    if not os.path.exists(model_path) and os.path.exists("deployment/model.pkl"):
        model_path = "deployment/model.pkl"
    if not os.path.exists(preprocessor_path) and os.path.exists("deployment/preprocessor.pkl"):
        preprocessor_path = "deployment/preprocessor.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"[INFO] Successfully loaded model from {model_path}")
    print(f"[INFO] Successfully loaded preprocessor from {preprocessor_path}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class HouseFeatureInput(BaseModel):
    MSSubClass: Union[int, str] = 20
    LotArea: float = Field(default=8450, ge=0)
    OverallQual: int = Field(default=7, ge=1, le=10)
    OverallCond: int = Field(default=5, ge=1, le=10)
    YearBuilt: int = Field(default=2003, ge=1800, le=2030)
    YearRemodAdd: int = Field(default=2003, ge=1800, le=2030)
    GrLivArea: float = Field(default=1710, ge=0)
    TotalBsmtSF: float = Field(default=856, ge=0)
    FullBath: int = Field(default=2, ge=0)
    HalfBath: int = Field(default=1, ge=0)
    BedroomAbvGr: int = Field(default=3, ge=0)
    TotRmsAbvGrd: int = Field(default=8, ge=0)
    Fireplaces: int = Field(default=0, ge=0)
    GarageCars: float = Field(default=2, ge=0)
    GarageArea: float = Field(default=548, ge=0)
    Neighborhood: str = "CollgCr"
    HouseStyle: str = "2Story"
    CentralAir: str = "Y"
    SaleCondition: str = "Normal"


class HousePricePrediction(BaseModel):
    predicted_price_usd: float
    log_predicted_price: float
    model_version: str = "1.0"
    status: str = "success"


@app.get("/")
def root():
    return {
        "service": "Ames House Price Prediction API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None and preprocessor is not None else "degraded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict", response_model=HousePricePrediction)
def predict(features: Dict[str, Any]):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    try:
        # 1. Convert to DataFrame
        input_df = pd.DataFrame([features])

        # 2. Engineer features matching training pipeline
        engineered = engineer_features(input_df)

        # 3. Transform via fitted ColumnTransformer
        X_proc = preprocessor.transform(engineered)

        # 4. Predict (log1p scale)
        log_pred = float(model.predict(X_proc)[0])

        # 5. Invert log1p to actual dollars
        pred_price = float(np.expm1(log_pred))

        return HousePricePrediction(
            predicted_price_usd=round(pred_price, 2),
            log_predicted_price=round(log_pred, 4),
            model_version="Production_Ridge_v1",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
def batch_predict(batch: List[Dict[str, Any]]):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    if not batch:
        raise HTTPException(status_code=400, detail="Empty batch submitted.")

    try:
        input_df = pd.DataFrame(batch)
        engineered = engineer_features(input_df)
        X_proc = preprocessor.transform(engineered)
        log_preds = model.predict(X_proc)
        prices = np.expm1(log_preds)

        results = []
        for i, (lp, p) in enumerate(zip(log_preds, prices)):
            results.append({
                "record_index": i,
                "predicted_price_usd": round(float(p), 2),
                "log_predicted_price": round(float(lp), 4)
            })

        return {
            "count": len(results),
            "predictions": results,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
