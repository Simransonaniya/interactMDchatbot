"""
Pydantic Schemas for InteractMD API.
Includes User Auth, Cases, Sessions, Chat Simulation, and Evaluation schemas.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

# ==========================================
# AUTH & USER SCHEMAS
# ==========================================

class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    role: Optional[str] = "LEARNER"

class UserCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")
    role: Optional[str] = "LEARNER"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None


# ==========================================
# CLINICAL ENTITY SCHEMAS
# ==========================================

class PatientProfileSchema(BaseModel):
    name: str
    age: int
    gender: str
    occupation: Optional[str] = None
    persona: Optional[str] = None

    class Config:
        from_attributes = True

class PhysicalFindingSchema(BaseModel):
    id: str
    system: str
    finding: str
    value: str
    description: Optional[str] = None

    class Config:
        from_attributes = True

class InvestigationSchema(BaseModel):
    id: str
    name: str
    category: str
    result: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_available: bool = True

    class Config:
        from_attributes = True

class ClinicalFactSchema(BaseModel):
    chief_complaint: Optional[str] = None
    history: Optional[str] = None
    onset: Optional[str] = None
    timing: Optional[str] = None
    location: Optional[str] = None
    character: Optional[str] = None
    severity: Optional[str] = None
    radiation: Optional[str] = None
    aggravating_factors: Optional[str] = None
    relieving_factors: Optional[str] = None
    associated_symptoms: Optional[List[str]] = None
    past_medical_history: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    facts_json: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# ==========================================
# CASE SCHEMAS
# ==========================================

class CaseSummary(BaseModel):
    id: str
    title: str
    specialty: str
    description: str
    difficulty: str
    is_published: bool
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class CaseDetail(BaseModel):
    id: str
    title: str
    specialty: str
    description: str
    difficulty: str
    is_published: bool
    patient: Optional[PatientProfileSchema] = None
    physical_findings: List[PhysicalFindingSchema] = []
    investigations: List[InvestigationSchema] = []
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# ADMIN CASE CREATION SCHEMA
# ==========================================

class AdminCaseCreate(BaseModel):
    title: str
    specialty: str
    description: str
    difficulty: Optional[str] = "Intermediate"
    is_published: Optional[bool] = True
    
    # Patient Profile
    patient_name: str
    patient_age: int
    patient_gender: str
    patient_occupation: Optional[str] = None
    patient_persona: Optional[str] = None

    # Clinical Facts
    chief_complaint: Optional[str] = None
    history: Optional[str] = None
    onset: Optional[str] = None
    timing: Optional[str] = None
    location: Optional[str] = None
    character: Optional[str] = None
    severity: Optional[str] = None
    radiation: Optional[str] = None
    aggravating_factors: Optional[str] = None
    relieving_factors: Optional[str] = None
    associated_symptoms: Optional[List[str]] = None
    past_medical_history: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    facts_json: Optional[Dict[str, Any]] = None

    # Physical Findings
    physical_findings: Optional[List[Dict[str, Any]]] = None

    # Investigations
    investigations: Optional[List[Dict[str, Any]]] = None

    # Hidden Evaluation Data
    diagnosis: str
    differential_diagnosis: Optional[List[str]] = None
    management: Optional[List[str]] = None
    learning_objectives: Optional[List[str]] = None
    scoring_rubric: Optional[Dict[str, Any]] = None


# ==========================================
# SIMULATION SESSION & MESSAGE SCHEMAS
# ==========================================

class StartSessionRequest(BaseModel):
    case_id: str

class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    case_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class MessageResponse(BaseModel):
    id: str
    session_id: str
    sender: str
    message: str
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# CHAT SIMULATION & INTERACTION SCHEMAS
# ==========================================

class HistoryMessage(BaseModel):
    id: Optional[str] = None
    sender: str
    text: str
    category: Optional[str] = "General"
    empathyDetected: Optional[bool] = False

class ChatRequest(BaseModel):
    case_id: str
    message: str
    session_id: Optional[str] = None
    conversation_history: List[HistoryMessage] = []

class ChatResponse(BaseModel):
    reply: str
    empathy_detected: bool
    category: str
    provider: str
    suggested_topics: List[str] = []
    session_id: Optional[str] = None

class ExamRequest(BaseModel):
    case_id: str
    exam_id: Optional[str] = None
    system: Optional[str] = None

class InvestigationRequest(BaseModel):
    case_id: str
    test_id: str


# ==========================================
# EVALUATION & OSCE SCHEMAS
# ==========================================

class EvaluationRequest(BaseModel):
    case_id: str
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []
    performed_exam_ids: List[str] = []
    ordered_investigation_ids: List[str] = []
    primary_diagnosis_id: str = ""
    differential_diagnosis_ids: List[str] = []
    selected_management_ids: List[str] = []
    clinical_rationale: str = ""
    duration_seconds: int = 0

class DimensionScore(BaseModel):
    dimension: str
    score: int
    feedback: str
    highValueHits: Optional[List[str]] = []

class EvaluationResponse(BaseModel):
    session_id: Optional[str] = None
    case_id: str
    overall_score: int
    pass_status: bool
    dimensions: List[DimensionScore]
    strengths: List[str]
    areas_to_improve: List[str]
    attending_physician_notes: str
    rubric_breakdown: Optional[Dict[str, Any]] = None
