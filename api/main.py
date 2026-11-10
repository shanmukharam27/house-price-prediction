import time
import uuid
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import HouseInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from api.monitoring_logger import log_inference

model_loader = ModelLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load()
    yield
    print("[INFO] House Price Prediction API shutting down...")

app = FastAPI(
    title="House Price Prediction API",
    description="Lab 10 & 11 - ML Model Serving and API Engineering for Ames Housing Regression",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)
        errors.append({"field": field, "message": error["msg"]})

    return JSONResponse(status_code=422, content={"status": "error", "message": "Validation failed", "details": errors})

@app.get("/")
def read_root():
    return {
        "service": "Ames House Price Prediction API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loader.model is not None and model_loader.preprocessor is not None else "degraded",
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(house: dict):
    start_time = time.time()
    try:
        df = pd.DataFrame([house])
        df_clean = clean_prediction_data(df)
        X_trans = model_loader.preprocessor.transform(df_clean)
        
        log_pred = float(model_loader.model.predict(X_trans)[0])
        pred_price = float(np.expm1(log_pred))
        
        latency_ms = (time.time() - start_time) * 1000
        log_inference(
            input_data=house,
            latency_ms=latency_ms,
            status="success",
            error=None,
            model_name=getattr(model_loader, "model_name", "HousingRidgeModel"),
            prediction=pred_price
        )

        return PredictionResponse(
            predicted_price_usd=round(pred_price, 2),
            log_predicted_price=round(log_pred, 4),
            model_name=getattr(model_loader, "model_name", "HousingRidgeModel"),
            status="success"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
