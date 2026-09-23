"""
Cases API Router.
GET  /api/v1/cases
GET  /api/v1/cases/{case_id}
POST /api/v1/admin/cases (Admin only)
"""

import uuid
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from schemas import CaseSummary, CaseDetail, AdminCaseCreate
from services.case_service import (
    get_all_cases,
    get_case_by_id,
    format_case_summary,
    format_case_detail,
    create_admin_case
)
from api.deps import get_current_admin_user, get_current_user_optional
from mongo_db import mongo_manager

router = APIRouter(tags=["Cases"])


@router.get("/api/v1/cases", response_model=List[CaseSummary])
@router.get("/api/cases", response_model=List[CaseSummary])
def list_cases(specialty: Optional[str] = None, db: Session = Depends(get_db)):
    """
    List all available clinical simulation cases from MongoDB Atlas (source of truth).
    """
    # 1. Fetch from MongoDB Atlas
    mongo_cases = mongo_manager.get_all_cases(specialty=specialty)
    if mongo_cases:
        return [format_case_summary(c) for c in mongo_cases]

    # 2. Fallback to SQL DB if available
    try:
        sql_cases = get_all_cases(db, specialty=specialty)
        if sql_cases:
            return [format_case_summary(c) for c in sql_cases]
    except Exception:
        pass

    return []


@router.get("/api/v1/cases/{case_id}", response_model=CaseDetail)
@router.get("/api/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, db: Session = Depends(get_db)):
    """
    Get full details for clinical simulation (Patient profile, physical findings, investigations).
    Sensitive ground truth diagnosis and scoring rubric are NEVER included in this response.
    """
    # 1. Fetch from MongoDB Atlas
    case = mongo_manager.get_case_by_id(case_id)

    # 2. Fallback to SQL DB if not found
    if not case:
        try:
            case = get_case_by_id(db, case_id)
        except Exception:
            pass

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical case '{case_id}' was not found in the database."
        )
    return format_case_detail(case)


@router.post("/api/v1/admin/cases", response_model=CaseSummary, status_code=status.HTTP_201_CREATED)
@router.post("/api/admin/cases", response_model=CaseSummary, status_code=status.HTTP_201_CREATED)
@router.post("/api/v1/cases", response_model=CaseSummary, status_code=status.HTTP_201_CREATED)
def admin_create_case(
    case_in: AdminCaseCreate,
    db: Session = Depends(get_db),
    admin_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Admin-only endpoint to create a new clinical case in MongoDB Atlas.
    """
    case_dict = case_in.model_dump()
    case_id = f"CS-{uuid.uuid4().hex[:6].upper()}" if "uuid" in globals() else f"case_{int(time.time())}"
    case_dict["case_id"] = case_id
    case_dict["id"] = case_id
    case_dict["status"] = "published"
    case_dict["is_published"] = case_in.is_published if case_in.is_published is not None else True
    case_dict["patient"] = {
        "name": case_in.patient_name,
        "age": case_in.patient_age,
        "gender": case_in.patient_gender,
        "occupation": case_in.patient_occupation,
        "persona": case_in.patient_persona
    }
    case_dict["examination"] = case_in.physical_findings or []
    case_dict["investigations"] = case_in.investigations or []
    case_dict["diagnosis_model"] = {
        "primary_diagnosis": case_in.diagnosis,
        "differentials": case_in.differential_diagnosis or []
    }
    case_dict["scoring_rubric"] = case_in.scoring_rubric or {}

    saved_id = mongo_manager.save_case(case_dict)
    
    try:
        create_admin_case(db, case_in)
    except Exception:
        pass

    return format_case_summary(case_dict)

