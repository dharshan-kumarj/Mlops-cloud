"""
Spam Classifier — FastAPI Application
======================================
Wraps the trained ensemble model in a REST API.
Every prediction is saved to PostgreSQL.
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Prediction, get_db, init_db

STATIC_DIR = Path(__file__).parent / "static"

# ──────────────────────────────────────────────
# Model loading at startup
# ──────────────────────────────────────────────
ENSEMBLE_PATH = "./ensemble.pkl"
VECTORIZER_PATH = "./vectorizer.pkl"
LABELS = {0: "ham", 1: "spam"}

model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup and init DB."""
    global model, vectorizer

    # Load model artifacts
    if not os.path.isfile(ENSEMBLE_PATH):
        raise RuntimeError(f"Model not found: {ENSEMBLE_PATH}. Run train.py first.")
    if not os.path.isfile(VECTORIZER_PATH):
        raise RuntimeError(f"Vectorizer not found: {VECTORIZER_PATH}. Run train.py first.")

    model = joblib.load(ENSEMBLE_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Create DB tables
    init_db()

    print("✓ Models loaded, database ready.")
    yield


app = FastAPI(
    title="Spam Classifier API",
    description="Classify SMS messages as spam or ham using a calibrated ensemble model. "
                "All predictions are persisted to PostgreSQL.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend UI."""
    return FileResponse(STATIC_DIR / "index.html")


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "Congratulations! You won a free prize!"}
        }


class PredictResponse(BaseModel):
    input_text: str
    prediction: str
    confidence: float
    ham_probability: float
    spam_probability: float
    id: int


class PredictionRecord(BaseModel):
    id: int
    input_text: str
    prediction: str
    confidence: float
    spam_probability: float
    ham_probability: float
    created_at: str

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Same preprocessing as training."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def classify(text: str) -> dict:
    """Run prediction and return results dict."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "prediction": LABELS[pred],
        "confidence": round(float(proba.max()) * 100, 2),
        "ham_probability": round(float(proba[0]), 4),
        "spam_probability": round(float(proba[1]), 4),
    }


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Classify a message as spam or ham.
    The prediction is automatically saved to PostgreSQL.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = classify(req.text)

    # Save to DB
    record = Prediction(
        input_text=result["input_text"],
        cleaned_text=result["cleaned_text"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        spam_probability=result["spam_probability"],
        ham_probability=result["ham_probability"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {**result, "id": record.id}


@app.get("/predictions", response_model=list[PredictionRecord])
def list_predictions(
    label: Optional[str] = Query(None, description="Filter by 'spam' or 'ham'"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    List saved predictions. Optionally filter by label (spam/ham).
    """
    query = db.query(Prediction).order_by(Prediction.created_at.desc())
    if label:
        if label not in ("spam", "ham"):
            raise HTTPException(status_code=400, detail="Label must be 'spam' or 'ham'.")
        query = query.filter(Prediction.prediction == label)
    records = query.limit(limit).all()

    return [
        PredictionRecord(
            id=r.id,
            input_text=r.input_text,
            prediction=r.prediction,
            confidence=r.confidence,
            spam_probability=r.spam_probability,
            ham_probability=r.ham_probability,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]


@app.get("/predictions/stats")
def prediction_stats(db: Session = Depends(get_db)):
    """
    Get summary stats: total predictions, spam count, ham count.
    """
    total = db.query(Prediction).count()
    spam_count = db.query(Prediction).filter(Prediction.prediction == "spam").count()
    ham_count = total - spam_count

    return {
        "total_predictions": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
    }


@app.get("/health")
def health():
    """Health check — confirms model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }
