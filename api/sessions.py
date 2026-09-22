"""
Simulation Sessions & Messaging API Router.
POST /api/v1/sessions
GET  /api/v1/sessions
GET  /api/v1/sessions/{session_id}
POST /api/v1/sessions/{session_id}/messages
GET  /api/v1/sessions/{session_id}/messages
"""

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from models.session_model import SimulationSession
from models.case_model import Case
from models.message import Message
from schemas import (
    StartSessionRequest,
    SessionResponse,
    SendMessageRequest,
    MessageResponse
)
from api.deps import get_current_user, get_current_user_optional
from services.case_service import get_case_by_id
from ai_engine import AIPatientEngine

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])
ai_engine = AIPatientEngine()


def build_case_dict_for_ai(case: Case) -> dict:
    """Extracts clinical facts and patient persona from DB models into AI engine format."""
    facts = case.clinical_facts
    patient = case.patient_profile

    return {
        "case_id": case.id,
        "patient": {
            "name": patient.name if patient else "Patient",
            "age": patient.age if patient else 45,
            "gender": patient.gender if patient else "Unknown",
            "occupation": patient.occupation if patient else "",
            "presentationComplaint": facts.chief_complaint if facts and facts.chief_complaint else case.description,
            "persona": patient.persona if patient else ""
        },
        "facts": {
            "chiefComplaint": facts.chief_complaint if facts else case.description,
            "history": facts.history if facts else "",
            "onset": facts.onset if facts else "It started recently.",
            "timing": facts.timing if facts else "It has been continuous.",
            "location": facts.location if facts else "In the central area.",
            "character": facts.character if facts else "A strong uncomfortable sensation.",
            "quality": facts.character if facts else "An uncomfortable sensation.",
            "severity": facts.severity if facts else "Quite severe.",
            "radiation": facts.radiation if facts else None,
            "provocationPalliative": f"Aggravated by: {facts.aggravating_factors or 'activity'}. Relieved by: {facts.relieving_factors or 'nothing so far'}.",
            "associatedSymptoms": facts.associated_symptoms if facts and facts.associated_symptoms else [],
            "pastMedicalHistory": facts.past_medical_history if facts and facts.past_medical_history else [],
            "medications": facts.medications if facts and facts.medications else [],
            "allergies": facts.allergies if facts and facts.allergies else [],
            "familyHistory": facts.family_history if facts else "No known significant family history.",
            "socialHistory": facts.social_history if facts else "Non-smoker, drinks socially.",
        }
    }


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: StartSessionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Start a new patient simulation session."""
    case = get_case_by_id(db, body.case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical case with ID '{body.case_id}' was not found in the database."
        )
    
    session = SimulationSession(
        id=str(uuid.uuid4()),
        user_id=current_user.id if current_user else None,
        case_id=body.case_id,
        status="ACTIVE"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse.model_validate(session)


@router.get("", response_model=List[SessionResponse])
def get_user_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all past and active simulation sessions for the current authenticated user.
    STRICT ISOLATION: A user can NEVER view another user's sessions.
    """
    sessions = db.query(SimulationSession).filter(
        SimulationSession.user_id == current_user.id
    ).order_by(SimulationSession.created_at.desc()).all()
    
    return [SessionResponse.model_validate(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session_detail(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get single session details with ownership verification."""
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user and session.user_id and session.user_id != current_user.id and current_user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    return SessionResponse.model_validate(session)


@router.post("/{session_id}/messages", response_model=MessageResponse)
def send_message(
    session_id: str,
    body: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Learner sends a dialogue turn to the AI patient in a simulation session.
    1. Authenticates & verifies session ownership.
    2. Loads selected case from PostgreSQL.
    3. Loads conversation history.
    4. Generates focused, question-specific patient response.
    5. Saves both messages to PostgreSQL.
    6. Returns patient response.
    """
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    
    # Verify ownership
    if current_user and session.user_id and session.user_id != current_user.id and current_user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    case = get_case_by_id(db, session.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case associated with this session no longer exists.")

    # 1. Fetch previous messages for context
    past_messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).all()

    conv_history = [
        {
            "sender": m.sender,
            "text": m.message,
            "category": (m.metadata_json or {}).get("category")
        }
        for m in past_messages
    ]

    # 2. Build case data dictionary from PostgreSQL
    case_ai_data = build_case_dict_for_ai(case)

    # 3. Generate patient response
    turn_result = ai_engine.process_turn(
        case_data=case_ai_data,
        user_message=body.message,
        conversation_history=conv_history
    )

    # 4. Save learner message in DB
    learner_msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        sender="LEARNER",
        message=body.message,
        metadata_json={"empathy_detected": turn_result["empathy_detected"]}
    )
    db.add(learner_msg)

    # 5. Save patient response in DB
    patient_msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        sender="PATIENT",
        message=turn_result["reply"],
        metadata_json={
            "category": turn_result["category"],
            "provider": turn_result["provider"],
            "suggested_topics": turn_result.get("suggested_topics", [])
        }
    )
    db.add(patient_msg)
    db.commit()
    db.refresh(patient_msg)

    return MessageResponse.model_validate(patient_msg)


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get all messages for a session.
    Strict ownership verification ensures User B cannot read User A's dialogue.
    """
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    
    if current_user and session.user_id and session.user_id != current_user.id and current_user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session's messages.")

    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).all()

    return [MessageResponse.model_validate(m) for m in messages]
