"""
Simulation Sessions & Messaging API Router.
Authoritative session persistence in MongoDB Atlas with user isolation,
progressive fact disclosure, and specialized clinical engines.
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from schemas import (
    StartSessionRequest,
    SessionResponse,
    SendMessageRequest,
    MessageResponse,
    ExamRequest,
    ExamResponse,
    InvestigationRequest,
    InvestigationResponse,
    DiagnosisSubmissionRequest,
    DiagnosisSubmissionResponse,
    ManagementSubmissionRequest,
    ManagementSubmissionResponse,
    EvaluationRequest,
    EvaluationResponse
)
from api.deps import get_current_user, get_current_user_optional
from mongo_db import mongo_manager
from ai_orchestrator import ai_orchestrator
from services.examination_service import examination_service
from services.investigation_service import investigation_service
from services.clinical_submission_service import clinical_submission_service
from services.evaluation_service import evaluation_service

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


def _format_session_response(session_doc: Dict[str, Any]) -> SessionResponse:
    s_id = session_doc.get("session_id") or session_doc.get("id", "")
    created = session_doc.get("created_at") or session_doc.get("started_at") or time.time()
    started = session_doc.get("started_at") or created
    updated = session_doc.get("updated_at") or created
    completed = session_doc.get("completed_at") or session_doc.get("ended_at")

    def _to_dt(val):
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        if isinstance(val, datetime):
            return val
        return datetime.now(timezone.utc)

    return SessionResponse(
        id=s_id,
        user_id=session_doc.get("user_id"),
        case_id=session_doc.get("case_id", ""),
        status=session_doc.get("status", "ACTIVE"),
        started_at=_to_dt(started),
        completed_at=_to_dt(completed) if completed else None,
        created_at=_to_dt(created),
        updated_at=_to_dt(updated)
    )


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: StartSessionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Start a new patient simulation session in MongoDB."""
    case = mongo_manager.get_case_by_id(body.case_id)
    if not case:
        all_cases = mongo_manager.get_all_cases()
        matched = next((c for c in all_cases if c.get("case_id") == body.case_id or c.get("id") == body.case_id), None)
        if matched:
            case = matched
        elif all_cases:
            case = all_cases[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Clinical case with ID '{body.case_id}' was not found in MongoDB."
            )

    case_id = case.get("case_id") or case.get("id") or body.case_id
    case_version = case.get("version", 1)

    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "id": session_id,
        "user_id": current_user.id if current_user else None,
        "case_id": case_id,
        "case_version": case_version,
        "status": "ACTIVE",
        "started_at": time.time(),
        "created_at": time.time(),
        "updated_at": time.time(),
        "revealed_fact_ids": [],
        "completed_examinations": [],
        "ordered_investigations": [],
        "revealed_investigation_results": [],
        "current_patient_state": "stable"
    }

    mongo_manager.create_session(session_data)
    mongo_manager.save_event(session_id, "SESSION_STARTED", {"case_id": case_id, "case_version": case_version})

    return _format_session_response(session_data)


@router.get("", response_model=List[SessionResponse])
def get_user_sessions(
    current_user: User = Depends(get_current_user)
):
    """
    Get all past and active simulation sessions for the current authenticated user.
    STRICT ISOLATION: A user can NEVER view another user's sessions.
    """
    sessions = mongo_manager.get_user_sessions(current_user.id)
    return [_format_session_response(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session_detail(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get single session details with strict user isolation."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user and session.get("user_id") and session.get("user_id") != current_user.id and getattr(current_user, "role", "") != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    return _format_session_response(session)


@router.post("/{session_id}/messages", response_model=MessageResponse)
def send_message(
    session_id: str,
    body: SendMessageRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Learner sends a dialogue turn to the AI patient in a simulation session.
    1. Verifies session ownership.
    2. Loads selected case from MongoDB.
    3. Runs AI Orchestrator with progressive disclosure and fact grounding.
    4. Persists learner and patient messages in MongoDB.
    5. Returns structured patient response.
    """
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Simulation session not found.")

    if current_user and session.get("user_id") and session.get("user_id") != current_user.id and getattr(current_user, "role", "") != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    case_id = session.get("case_id", "chest_pain_001")

    # Generate patient response via AI Orchestrator
    turn_result = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message=body.message,
        session_id=session_id
    )

    created_dt = datetime.now(timezone.utc)
    return MessageResponse(
        id=str(uuid.uuid4()),
        session_id=session_id,
        sender="PATIENT",
        message=turn_result.get("reply", ""),
        metadata_json={
            "category": turn_result.get("category", "General"),
            "provider": turn_result.get("provider", "InteractMD AI Patient"),
            "empathy_detected": turn_result.get("empathy_detected", False),
            "facts_revealed": turn_result.get("facts_revealed", [])
        },
        created_at=created_dt
    )


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get all messages for a session from MongoDB.
    Strict ownership verification ensures User B cannot read User A's dialogue.
    """
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Simulation session not found.")

    if current_user and session.get("user_id") and session.get("user_id") != current_user.id and getattr(current_user, "role", "") != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session's messages.")

    raw_msgs = mongo_manager.get_session_messages(session_id)
    out = []
    for m in raw_msgs:
        t = m.get("timestamp") or time.time()
        dt = datetime.fromtimestamp(t, tz=timezone.utc) if isinstance(t, (int, float)) else datetime.now(timezone.utc)
        out.append(MessageResponse(
            id=m.get("id", str(uuid.uuid4())),
            session_id=session_id,
            sender=m.get("sender") or m.get("role", "learner").upper(),
            message=m.get("message") or m.get("content", ""),
            metadata_json=m.get("metadata_json") or m.get("metadata") or {},
            created_at=dt
        ))
    return out


# --------------------------------------------------------------------------
# Physical Examination Endpoint
# --------------------------------------------------------------------------
@router.post("/{session_id}/examinations")
@router.post("/{session_id}/examination")
def perform_session_examination(
    session_id: str,
    body: ExamRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Perform physical examination maneuver and return controlled MongoDB finding."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    case_id = body.case_id or session.get("case_id", "")
    return examination_service.perform_exam(
        case_id=case_id,
        session_id=session_id,
        exam_id=body.exam_id,
        system=body.system
    )


# --------------------------------------------------------------------------
# Diagnostic Investigations Endpoint
# --------------------------------------------------------------------------
@router.post("/{session_id}/investigations")
def order_session_investigation(
    session_id: str,
    body: InvestigationRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Order diagnostic test and return authoritative result from MongoDB case."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    case_id = body.case_id or session.get("case_id", "")
    test_id = body.test_id or body.test or "STAT 12-Lead ECG"
    return investigation_service.order_investigation(
        case_id=case_id,
        test_id=test_id,
        session_id=session_id
    )


# --------------------------------------------------------------------------
# Diagnosis Submission Endpoint
# --------------------------------------------------------------------------
@router.post("/{session_id}/diagnosis", response_model=DiagnosisSubmissionResponse)
def submit_session_diagnosis(
    session_id: str,
    body: DiagnosisSubmissionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Submit learner differential and primary working diagnosis."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    case_id = session.get("case_id", "")
    res = clinical_submission_service.submit_diagnosis(
        session_id=session_id,
        case_id=case_id,
        differential=body.differential,
        most_likely=body.most_likely,
        reasoning=body.reasoning
    )
    return DiagnosisSubmissionResponse(**res)


# --------------------------------------------------------------------------
# Management Submission Endpoint
# --------------------------------------------------------------------------
@router.post("/{session_id}/management", response_model=ManagementSubmissionResponse)
def submit_session_management(
    session_id: str,
    body: ManagementSubmissionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Submit clinical management, immediate actions, and escalation plan."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    case_id = session.get("case_id", "")
    res = clinical_submission_service.submit_management(
        session_id=session_id,
        case_id=case_id,
        immediate_actions=body.immediate_actions,
        treatment=body.treatment,
        escalation=body.escalation,
        investigations=body.investigations,
        selected_protocol_ids=body.selected_protocol_ids
    )
    return ManagementSubmissionResponse(**res)


# --------------------------------------------------------------------------
# Encounter Complete & Evaluation Endpoints
# --------------------------------------------------------------------------
@router.post("/{session_id}/complete", response_model=EvaluationResponse)
@router.post("/{session_id}/evaluate", response_model=EvaluationResponse)
def complete_session_and_evaluate(
    session_id: str,
    body: Optional[EvaluationRequest] = None,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Complete encounter and generate OSCE evaluation."""
    session = mongo_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    case_id = (body.case_id if body else None) or session.get("case_id", "")
    diag_sub = session.get("diagnosis_submission", {})
    mgmt_sub = session.get("management_submission", {})

    performed_exams = (body.performed_exam_ids if body and body.performed_exam_ids is not None else None) or session.get("completed_examinations", [])
    ordered_invs = (body.ordered_investigation_ids if body and body.ordered_investigation_ids is not None else None) or session.get("ordered_investigations", [])
    primary_dx = (body.primary_diagnosis_id if body and body.primary_diagnosis_id else None) or diag_sub.get("most_likely", "")
    diff_dx = (body.differential_diagnosis_ids if body and body.differential_diagnosis_ids is not None else None) or diag_sub.get("differential", [])
    mgmt_ids = (body.selected_management_ids if body and body.selected_management_ids is not None else None) or mgmt_sub.get("selected_protocol_ids", []) or mgmt_sub.get("immediate_actions", [])
    rationale = (body.clinical_rationale if body and body.clinical_rationale else None) or diag_sub.get("reasoning", "")
    duration = (body.duration_seconds if body and body.duration_seconds is not None else None) or int(time.time() - session.get("started_at", time.time()))

    eval_res = evaluation_service.evaluate_session(
        case_id=case_id,
        session_id=session_id,
        conversation_history=body.conversation_history if body else None,
        performed_exam_ids=performed_exams,
        ordered_investigation_ids=ordered_invs,
        primary_diagnosis_id=primary_dx,
        differential_diagnosis_ids=diff_dx,
        selected_management_ids=mgmt_ids,
        clinical_rationale=rationale,
        duration_seconds=duration
    )
    return eval_res


@router.get("/{session_id}/evaluation", response_model=Optional[EvaluationResponse])
def get_session_evaluation(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get completed evaluation for this session."""
    eval_doc = mongo_manager.get_evaluation(session_id)
    if not eval_doc:
        raise HTTPException(status_code=404, detail="Evaluation not yet generated for this session.")
    return EvaluationResponse.model_validate(eval_doc)

