"""
InteractMD — Python AI Patient Chatbot Backend with Full PostgreSQL DB & Auth Support.
Built with FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL, and JWT.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import settings
from database import engine as db_engine, Base, get_db
import models # ensure all models are registered
from schemas import (
    ChatRequest, ChatResponse,
    EvaluationRequest, EvaluationResponse,
    ExamRequest, InvestigationRequest
)
from ai_engine import AIPatientEngine
from evaluator import evaluate_encounter
from services.case_service import get_case_by_id

from api.auth import router as auth_router
from api.cases import router as cases_router
from api.sessions import router as sessions_router
from models.session_model import SimulationSession
from models.message import Message
from models.evaluation import Evaluation

from mongo_db import mongo_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure Database and MongoDB Atlas connections
    try:
        Base.metadata.create_all(bind=db_engine)
        print("[Lifespan] Relational tables checked.")
    except Exception as e:
        print(f"[Lifespan Warning] Relational check: {e}")

    try:
        if mongo_manager.is_connected:
            print("[Lifespan] MongoDB Atlas connected and active.")
    except Exception as e:
        print(f"[Lifespan Warning] MongoDB check: {e}")
    yield

app = FastAPI(
    title="InteractMD Python AI Backend",
    description="Full Python Backend for AI Clinical Simulation, Auth, Cases, & Session Persistence with RAG and MongoDB Atlas",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(cases_router)
app.include_router(sessions_router)

ai_patient_engine = AIPatientEngine()


@app.get("/")
def root():
    return {
        "service": "InteractMD Python Backend",
        "status": "online",
        "version": "2.0.0",
        "database": "MongoDB Atlas" if mongo_manager.is_connected else "Local/PostgreSQL",
        "rag_engine": "Active",
        "docs": "/docs"
    }


@app.get("/health")
def health_endpoint():
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_endpoint(db: Session = Depends(get_db)):
    """Readiness probe that tests database connectivity."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass

    if db_connected or mongo_manager.is_connected:
        return {
            "status": "ready",
            "database": "connected",
            "mongo_atlas": "connected" if mongo_manager.is_connected else "offline"
        }

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "database": "disconnected"
        }
    )


@app.get("/api/health")
def api_health_check():
    """Healthcheck endpoint called by the frontend (apiClient.ts)."""
    return {
        "status": "online",
        "provider": ai_patient_engine.provider_name,
        "rag": True,
        "database": "MongoDB Atlas" if mongo_manager.is_connected else "PostgreSQL"
    }


from ai_orchestrator import ai_orchestrator
from services.examination_service import examination_service
from services.investigation_service import investigation_service
from services.evaluation_service import evaluation_service


@app.post("/api/simulation/chat", response_model=ChatResponse)
def patient_chat(request: ChatRequest):
    """
    Core AI Patient dialogue turn.
    Authoritatively loads clinical ground truth from MongoDB.
    Runs AI Orchestrator with progressive fact disclosure and Response Validator.
    Persists dialogue and interaction events in MongoDB.
    """
    history_dicts = [m.model_dump() if hasattr(m, 'model_dump') else dict(m) for m in request.conversation_history]
    
    result = ai_orchestrator.process_turn_sync(
        case_id=request.case_id,
        user_message=request.message,
        session_id=request.session_id,
        conversation_history=history_dicts
    )

    return ChatResponse(
        session_id=result.get("session_id") or request.session_id,
        message={"role": "patient", "text": result.get("reply", "")},
        reply=result.get("reply", ""),
        empathy_detected=result.get("empathy_detected", False),
        category=result.get("category", "General"),
        provider=result.get("provider", ai_orchestrator.provider_name),
        facts_revealed=result.get("facts_revealed", []),
        suggested_topics=result.get("suggested_topics", []),
        session_state=result.get("session_state")
    )


@app.post("/api/simulation/evaluate", response_model=EvaluationResponse)
def submit_evaluation(request: EvaluationRequest):
    """
    Attending Physician OSCE evaluation endpoint.
    Scores the completed encounter across 5 dimensions against MongoDB scoring rubric.
    """
    return evaluation_service.evaluate_session(
        case_id=request.case_id,
        session_id=request.session_id,
        conversation_history=request.conversation_history,
        performed_exam_ids=request.performed_exam_ids,
        ordered_investigation_ids=request.ordered_investigation_ids,
        primary_diagnosis_id=request.primary_diagnosis_id,
        differential_diagnosis_ids=request.differential_diagnosis_ids,
        selected_management_ids=request.selected_management_ids,
        clinical_rationale=request.clinical_rationale,
        duration_seconds=request.duration_seconds
    )


@app.post("/api/simulation/exam")
def perform_exam(request: ExamRequest):
    """Physical examination maneuver results loaded authoritatively from MongoDB."""
    return examination_service.perform_exam(
        case_id=request.case_id or "chest_pain_001",
        session_id=None,
        exam_id=request.exam_id,
        system=request.system
    )


@app.post("/api/simulation/investigation")
def order_investigation(request: InvestigationRequest):
    """Diagnostic investigations loaded authoritatively from MongoDB."""
    test_id = request.test_id or request.test or "STAT 12-Lead ECG"
    return investigation_service.order_investigation(
        case_id=request.case_id or "chest_pain_001",
        test_id=test_id,
        session_id=None
    )

