"""FastAPI service for demand forecasting."""

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from features import FEATURE_COLUMNS
from production_model import (
    load_production_model,
    predict_demand,
)


app = FastAPI(
    title="E-Commerce Demand Forecasting API",
    description=(
        "Production inference API for daily product-level "
        "demand forecasting."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    product_id: str
    category: str

    price: float = Field(
        ge=0,
    )

    promotion: int = Field(
        ge=0,
        le=1,
    )

    discount: float = Field(
        ge=0,
    )

    holiday: int = Field(
        ge=0,
        le=1,
    )

    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_std_7: float

    rolling_mean_14: float
    rolling_std_14: float

    rolling_mean_28: float
    rolling_std_28: float

    day_of_week: int = Field(
        ge=0,
        le=6,
    )

    month: int = Field(
        ge=1,
        le=12,
    )

    week_of_year: int = Field(
        ge=1,
        le=53,
    )

    is_weekend: int = Field(
        ge=0,
        le=1,
    )


class PredictionResponse(BaseModel):
    product_id: str
    category: str
    predicted_demand: float


try:
    MODEL = load_production_model()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)


@app.get("/health")
def health_check():
    """Health check endpoint."""

    if MODEL is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model-info")
def model_info():
    """Return basic production model information."""

    return {
        "model_type": (
            type(MODEL).__name__
            if MODEL is not None
            else None
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """Predict daily demand for one product."""

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Production model is unavailable: "
                f"{MODEL_LOAD_ERROR}"
            ),
        )

    data = request.model_dump()

    product_id = data.pop("product_id")
    category = data.pop("category")

    df = pd.DataFrame([data])

    try:
        prediction = predict_demand(
            df,
            model=MODEL,
        )[0]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        product_id=product_id,
        category=category,
        predicted_demand=round(
            float(prediction),
            4,
        ),
    )