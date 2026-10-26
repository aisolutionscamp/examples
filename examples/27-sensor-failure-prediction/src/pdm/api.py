"""FastAPI scoring endpoint.

    MODEL_PATH=model.joblib uvicorn pdm.api:app --app-dir src --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import MaintenanceModel


class ScoreRequest(BaseModel):
    machine_id: str
    readings: list[dict[str, float]] = Field(description="one dict per cycle: cycle + every sensor")


class ScoreResponse(BaseModel):
    machine_id: str
    cycle: int
    failure_probability: float
    rul_estimate: float
    anomaly_score: float
    anomaly: bool
    alert: bool


def create_app(model: MaintenanceModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or MaintenanceModel.load(os.environ.get("MODEL_PATH", "model.joblib"))
        yield

    app = FastAPI(title="Predictive maintenance scoring", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        m: MaintenanceModel = app.state.model
        required = {"cycle", *m.sensors}
        if len(req.readings) < m.window:
            raise HTTPException(422, f"need at least {m.window} readings, got {len(req.readings)}")
        missing = required - set().union(*(r.keys() for r in req.readings))
        if missing:
            raise HTTPException(422, f"missing fields: {sorted(missing)}")
        df = pd.DataFrame(req.readings).assign(machine_id=req.machine_id)
        last = m.score(df).iloc[-1]  # most recent cycle
        return ScoreResponse(machine_id=req.machine_id, cycle=int(last["cycle"]),
                             failure_probability=float(last["failure_probability"]),
                             rul_estimate=float(last["rul_estimate"]),
                             anomaly_score=float(last["anomaly_score"]),
                             anomaly=bool(last["anomaly"]), alert=bool(last["alert"]))

    return app


app = create_app()  # the model is loaded from MODEL_PATH on startup
