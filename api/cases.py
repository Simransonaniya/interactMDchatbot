"""
Cases API Router.
GET  /api/v1/cases
GET  /api/v1/cases/{case_id}
POST /api/v1/admin/cases (Admin only)
"""

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
from api.deps import get_current_admin_user

router = APIRouter(tags=["Cases"])

@router.get("/api/v1/cases", response_model=List[CaseSummary])
@router.get("/api/cases", response_model=List[CaseSummary])
def list_cases(specialty: Optional[str] = None, db: Session = Depends(get_db)):
    """
    List all available clinical simulation cases from PostgreSQL.
    Returns empty list [] if database has 0 cases.
    """
    cases = get_all_cases(db, specialty=specialty)
    return [format_case_summary(c) for c in cases]

@router.get("/api/v1/cases/{case_id}", response_model=CaseDetail)
@router.get("/api/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, db: Session = Depends(get_db)):
    """
    Get full details for clinical simulation (Patient profile, physical findings, investigations).
    Sensitive ground truth diagnosis and scoring rubric are NEVER included in this response.
    """
    case = get_case_by_id(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical case '{case_id}' was not found in the database."
        )
    return format_case_detail(case)

@router.post("/api/v1/admin/cases", response_model=CaseSummary, status_code=status.HTTP_201_CREATED)
def admin_create_case(
    case_in: AdminCaseCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user)
):
    """
    Admin-only endpoint to create a new clinical case with patient profile,
    clinical facts, findings, investigations, and hidden evaluation rubrics.
    """
    new_case = create_admin_case(db, case_in)
    return format_case_summary(new_case)
