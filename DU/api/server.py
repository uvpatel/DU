"""FastAPI deployment helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Prediction input payload."""

    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    """Prediction output payload."""

    predictions: list[float | int | str]


def deploy_api(model: Any) -> FastAPI:
    """Create and return a FastAPI app with a /predict endpoint."""
    app = FastAPI(title="DU Model API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if not payload.records:
            raise HTTPException(status_code=400, detail="records cannot be empty")

        data = pd.DataFrame(payload.records)
        preds = model.predict(data)
        return PredictResponse(predictions=[p.item() if hasattr(p, "item") else p for p in preds])

    return app
