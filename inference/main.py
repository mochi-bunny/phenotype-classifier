"""
FastAPI backend — Agricultural AI Intelligence Platform.

Serves phenotype evaluation and disease diagnostics endpoints,
powered by a trained ONNX model (replaces the original LLM calls).
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from phenotype_detector import analyze_crop_phenotype
from disease_diagnostics import identify_crop_disease
from evaluator import filter_crop_by_preferences

app = FastAPI(
    title="Agricultural AI Intelligence Platform",
    version="2.0.0",
    description="Crop phenotyping and disease diagnostics powered by NVIDIA Cosmos + trained vision model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluationRequest(BaseModel):
    image_url: str = Field(..., description="Public URL or local path of the crop image")
    metadata: Dict[str, Any] = Field(default={}, description="Telemetry data (GPS, NDVI, soil moisture, etc.)")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="Optional target thresholds for trait filtering")


@app.post("/api/v1/evaluate/phenotype", summary="Run Phenotypic Evaluation & Trait Filtering")
def evaluate_phenotype_endpoint(payload: EvaluationRequest):
    try:
        ai_evaluation = analyze_crop_phenotype(payload.image_url, payload.metadata)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phenotype inference failed: {str(e)}")

    response_data = {
        "ai_evaluation": ai_evaluation,
        "preference_filtering": None,
    }

    if payload.user_preferences:
        response_data["preference_filtering"] = filter_crop_by_preferences(
            ai_evaluation, payload.user_preferences
        )

    return response_data


@app.post("/api/v1/evaluate/diagnostics", summary="Run Plant Pathology & Disease Diagnostics")
def evaluate_diagnostics_endpoint(payload: EvaluationRequest):
    try:
        ai_evaluation = identify_crop_disease(payload.image_url, payload.metadata)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease diagnostic inference failed: {str(e)}")

    return {"ai_evaluation": ai_evaluation}


@app.get("/health", summary="Health Check")
def health_check():
    model_status = "loaded"
    try:
        from phenotype_detector import _session
        if _session is None:
            model_status = "not_loaded"
    except Exception:
        model_status = "error"

    return {
        "status": "healthy",
        "service": "Agricultural AI Intelligence Platform API",
        "model_status": model_status,
    }


# Serve dashboard if present
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "dashboard.html")
if os.path.exists(DASHBOARD_PATH):
    @app.get("/", summary="Dashboard")
    def serve_dashboard():
        return FileResponse(DASHBOARD_PATH)
