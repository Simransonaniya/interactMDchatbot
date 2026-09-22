"""
Case management service.
Handles querying cases from PostgreSQL DB, creating cases via Admin endpoint,
and formatting cases for learner simulation (hiding evaluation data).
"""

import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from models.case_model import (
    Case,
    PatientProfile,
    ClinicalFact,
    PhysicalFinding,
    Investigation,
    CaseHiddenEvaluation
)
from schemas import AdminCaseCreate, CaseSummary, CaseDetail, PatientProfileSchema, PhysicalFindingSchema, InvestigationSchema

def get_all_cases(db: Session, specialty: Optional[str] = None) -> List[Case]:
    """
    Returns all published cases from PostgreSQL.
    Returns an empty list [] if no cases exist in DB.
    """
    query = db.query(Case).filter(Case.is_published == True)
    if specialty and specialty.lower() != "all":
        query = query.filter(Case.specialty.ilike(f"%{specialty}%"))
    return query.order_by(Case.created_at.desc()).all()

def get_case_by_id(db: Session, case_id: str) -> Optional[Case]:
    """Fetch single case by ID with related clinical entities."""
    return db.query(Case).filter(Case.id == case_id).first()

def format_case_summary(case: Case) -> CaseSummary:
    """Formats a database Case into a CaseSummary schema."""
    patient = case.patient_profile
    clinical_facts = case.clinical_facts
    
    return CaseSummary(
        id=case.id,
        title=case.title,
        specialty=case.specialty,
        description=case.description,
        difficulty=case.difficulty,
        is_published=case.is_published,
        patient_name=patient.name if patient else "Patient",
        patient_age=patient.age if patient else 45,
        patient_gender=patient.gender if patient else "Unknown",
        chief_complaint=clinical_facts.chief_complaint if clinical_facts and clinical_facts.chief_complaint else case.description,
        created_at=case.created_at
    )

def format_case_detail(case: Case) -> CaseDetail:
    """
    Formats a database Case into CaseDetail for learner simulation.
    CRITICAL: Does NOT include hidden diagnosis, differential, or scoring rubric.
    """
    patient = None
    if case.patient_profile:
        patient = PatientProfileSchema(
            name=case.patient_profile.name,
            age=case.patient_profile.age,
            gender=case.patient_profile.gender,
            occupation=case.patient_profile.occupation,
            persona=case.patient_profile.persona
        )

    physical_findings = [
        PhysicalFindingSchema(
            id=pf.id,
            system=pf.system,
            finding=pf.finding,
            value=pf.value,
            description=pf.description
        )
        for pf in (case.physical_findings or [])
    ]

    investigations = [
        InvestigationSchema(
            id=inv.id,
            name=inv.name,
            category=inv.category,
            result=inv.result,
            unit=inv.unit,
            reference_range=inv.reference_range,
            is_available=inv.is_available
        )
        for inv in (case.investigations or [])
    ]

    return CaseDetail(
        id=case.id,
        title=case.title,
        specialty=case.specialty,
        description=case.description,
        difficulty=case.difficulty,
        is_published=case.is_published,
        patient=patient,
        physical_findings=physical_findings,
        investigations=investigations,
        created_at=case.created_at
    )

def create_admin_case(db: Session, case_in: AdminCaseCreate) -> Case:
    """Creates a full clinical case with all related entities in PostgreSQL."""
    case_id = f"CS-{uuid.uuid4().hex[:6].upper()}"
    
    db_case = Case(
        id=case_id,
        title=case_in.title,
        specialty=case_in.specialty,
        description=case_in.description,
        difficulty=case_in.difficulty or "Intermediate",
        is_published=case_in.is_published if case_in.is_published is not None else True
    )
    db.add(db_case)
    db.flush()

    # 1. Patient Profile
    patient = PatientProfile(
        id=f"pt-{uuid.uuid4().hex[:8]}",
        case_id=case_id,
        name=case_in.patient_name,
        age=case_in.patient_age,
        gender=case_in.patient_gender,
        occupation=case_in.patient_occupation,
        persona=case_in.patient_persona
    )
    db.add(patient)

    # 2. Clinical Facts
    facts = ClinicalFact(
        id=f"cf-{uuid.uuid4().hex[:8]}",
        case_id=case_id,
        chief_complaint=case_in.chief_complaint,
        history=case_in.history,
        onset=case_in.onset,
        timing=case_in.timing,
        location=case_in.location,
        character=case_in.character,
        severity=case_in.severity,
        radiation=case_in.radiation,
        aggravating_factors=case_in.aggravating_factors,
        relieving_factors=case_in.relieving_factors,
        associated_symptoms=case_in.associated_symptoms or [],
        past_medical_history=case_in.past_medical_history or [],
        medications=case_in.medications or [],
        allergies=case_in.allergies or [],
        family_history=case_in.family_history,
        social_history=case_in.social_history,
        smoking=case_in.smoking,
        alcohol=case_in.alcohol,
        facts_json=case_in.facts_json or {}
    )
    db.add(facts)

    # 3. Physical Findings
    if case_in.physical_findings:
        for pf_dict in case_in.physical_findings:
            pf = PhysicalFinding(
                id=pf_dict.get("id") or f"pf-{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                system=pf_dict.get("system", "General"),
                finding=pf_dict.get("finding", ""),
                value=pf_dict.get("value", ""),
                description=pf_dict.get("description", "")
            )
            db.add(pf)

    # 4. Investigations
    if case_in.investigations:
        for inv_dict in case_in.investigations:
            inv = Investigation(
                id=inv_dict.get("id") or f"inv-{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                name=inv_dict.get("name", ""),
                category=inv_dict.get("category", "General"),
                result=inv_dict.get("result", ""),
                unit=inv_dict.get("unit"),
                reference_range=inv_dict.get("reference_range"),
                is_available=inv_dict.get("is_available", True)
            )
            db.add(inv)

    # 5. Hidden Evaluation Data (Never exposed to learner)
    hidden_eval = CaseHiddenEvaluation(
        id=f"che-{uuid.uuid4().hex[:8]}",
        case_id=case_id,
        diagnosis=case_in.diagnosis,
        differential_diagnosis=case_in.differential_diagnosis or [],
        management=case_in.management or [],
        learning_objectives=case_in.learning_objectives or [],
        scoring_rubric=case_in.scoring_rubric or {}
    )
    db.add(hidden_eval)

    db.commit()
    db.refresh(db_case)
    return db_case
