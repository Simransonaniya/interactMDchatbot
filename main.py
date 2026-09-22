"""
InteractMD — Python AI Patient Chatbot Backend with Full PostgreSQL DB & Auth Support.
Built with FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL, and JWT.
"""

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
from api.sessions import build_case_dict_for_ai

from api.auth import router as auth_router
from api.cases import router as cases_router
from api.sessions import router as sessions_router
from models.session_model import SimulationSession
from models.message import Message
from models.evaluation import Evaluation

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB connection can be initialized (without auto-seeding cases)
    try:
        Base.metadata.create_all(bind=db_engine)
        print("[Lifespan] Database initialized successfully. Zero default cases auto-seeded.")
    except Exception as e:
        print(f"[Lifespan Warning] Database check: {e}")
    yield

app = FastAPI(
    title="InteractMD Python AI Backend",
    description="Full Python Backend for AI Clinical Simulation, Auth, Cases, & Session Persistence",
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
        "docs": "/docs"
    }


@app.get("/health")
def health_endpoint():
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_endpoint(db: Session = Depends(get_db)):
    """Readiness probe that explicitly tests PostgreSQL database connectivity."""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected"
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "database": "disconnected",
                "detail": str(e)
            }
        )


@app.get("/api/health")
def api_health_check():
    """Healthcheck endpoint called by the frontend (apiClient.ts)."""
    return {
        "status": "online",
        "provider": ai_patient_engine.provider_name
    }


@app.post("/api/simulation/chat", response_model=ChatResponse)
def patient_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Core AI Patient dialogue turn.
    Called by frontend when learner asks the patient a question.
    Loads the case directly from PostgreSQL.
    """
    case = get_case_by_id(db, request.case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical case '{request.case_id}' was not found in the database."
        )

    case_dict = build_case_dict_for_ai(case)
    history_dicts = [m.model_dump() for m in request.conversation_history]
    
    result = ai_patient_engine.process_turn(
        case_data=case_dict,
        user_message=request.message,
        conversation_history=history_dicts
    )

    session_id = request.session_id
    if session_id:
        session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
        if session:
            user_msg = Message(
                session_id=session_id,
                sender="LEARNER",
                message=request.message,
                metadata_json={"empathy_detected": result["empathy_detected"]}
            )
            patient_msg = Message(
                session_id=session_id,
                sender="PATIENT",
                message=result["reply"],
                metadata_json={"category": result["category"], "provider": result["provider"]}
            )
            db.add_all([user_msg, patient_msg])
            db.commit()

    return ChatResponse(
        reply=result["reply"],
        empathy_detected=result["empathy_detected"],
        category=result["category"],
        provider=result["provider"],
        suggested_topics=result["suggested_topics"],
        session_id=session_id
    )


@app.post("/api/simulation/evaluate", response_model=EvaluationResponse)
def submit_evaluation(request: EvaluationRequest, db: Session = Depends(get_db)):
    """
    Attending Physician OSCE evaluation endpoint.
    Scores the completed encounter across 5 dimensions and persists evaluation to PostgreSQL.
    """
    case = get_case_by_id(db, request.case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical case '{request.case_id}' was not found in the database."
        )

    eval_resp = evaluate_encounter(request, case_obj=case)

    session_id = request.session_id or eval_resp.session_id
    if session_id:
        session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
        if session:
            session.status = "COMPLETED"
            
            existing_eval = db.query(Evaluation).filter(Evaluation.session_id == session_id).first()
            if not existing_eval:
                db_eval = Evaluation(
                    session_id=session_id,
                    user_id=session.user_id,
                    score=eval_resp.overall_score,
                    feedback=eval_resp.attending_physician_notes,
                    strengths=eval_resp.strengths,
                    areas_for_improvement=eval_resp.areas_to_improve,
                    category_scores={d.dimension: d.score for d in eval_resp.dimensions}
                )
                db.add(db_eval)
            db.commit()

    return eval_resp


@app.post("/api/simulation/exam")
def perform_exam(request: ExamRequest, db: Session = Depends(get_db)):
    """Physical examination maneuver results loaded from PostgreSQL."""
    case = get_case_by_id(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found in database.")
    
    findings = case.physical_findings or []
    if request.exam_id:
        matched = [f for f in findings if f.id == request.exam_id]
    elif request.system:
        matched = [f for f in findings if f.system.lower() == request.system.lower()]
    else:
        matched = findings

    return {
        "case_id": request.case_id,
        "findings": [
            {
                "id": f.id,
                "system": f.system,
                "finding": f.finding,
                "value": f.value,
                "description": f.description
            }
            for f in matched
        ]
    }


@app.post("/api/simulation/investigation")
def order_investigation(request: InvestigationRequest, db: Session = Depends(get_db)):
    """Diagnostic investigations loaded from PostgreSQL."""
    case = get_case_by_id(db, request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found in database.")

    investigations = case.investigations or []
    matched = next((inv for inv in investigations if inv.id.lower() == request.test_id.lower() or request.test_id.lower() in inv.name.lower()), None)
    
    if not matched:
        raise HTTPException(status_code=400, detail=f"Diagnostic test '{request.test_id}' not indicated or available.")

    return {
        "case_id": request.case_id,
        "test": matched.name,
        "turnaroundMinutes": 15,
        "result": {
            "id": matched.id,
            "name": matched.name,
            "category": matched.category,
            "result": matched.result,
            "unit": matched.unit,
            "reference_range": matched.reference_range,
            "is_available": matched.is_available
        }
    }
