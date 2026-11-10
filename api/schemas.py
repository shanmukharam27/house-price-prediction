from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class HouseInput(BaseModel):
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

    class Config:
        extra = "allow"

class PredictionResponse(BaseModel):
    predicted_price_usd: float
    log_predicted_price: float
    model_name: str = "Production_Housing_Model"
    status: str = "success"

class ErrorDetail(BaseModel):
    field: str
    message: str

class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]

class BatchPredictionRequest(BaseModel):
    houses: list[Dict[str, Any]] = Field(min_length=1)

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
