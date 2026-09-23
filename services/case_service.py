"""
Case management service.
Handles querying cases from PostgreSQL DB, creating cases via Admin endpoint,
and formatting cases for learner simulation (hiding evaluation data).
"""

import uuid
from datetime import datetime, timezone
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

def format_case_summary(case) -> CaseSummary:
    """Formats a database Case ORM model or MongoDB document into a CaseSummary schema."""
    if isinstance(case, dict):
        patient = case.get("patient") or {}
        facts = case.get("facts") or {}
        created = case.get("created_at")
        if isinstance(created, (int, float)):
            created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
        elif isinstance(created, datetime):
            created_dt = created
        elif isinstance(created, str):
            try:
                created_dt = datetime.fromisoformat(created)
            except Exception:
                created_dt = datetime.now(timezone.utc)
        else:
            created_dt = datetime.now(timezone.utc)

        avatar_url = patient.get("avatar_url") or patient.get("avatarUrl") or case.get("avatar_url")
        if not avatar_url:
            avatar_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80" if patient.get("gender") == "Male" else "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&auto=format&fit=crop&q=80"
        
        tags = case.get("tags") or [case.get("specialty", "General"), case.get("difficulty", "Intermediate"), "OSCE Simulation"]
        estimated_minutes = case.get("estimated_minutes") or case.get("estimatedMinutes") or 15

        initial_stmt = patient.get("initial_statement") or patient.get("initialStatement") or case.get("initial_statement") or case.get("opening_statement")

        return CaseSummary(
            id=case.get("id", ""),
            title=case.get("title", ""),
            specialty=case.get("specialty", "General"),
            description=case.get("description", ""),
            difficulty=case.get("difficulty", "Intermediate"),
            is_published=case.get("is_published", True),
            patient_name=patient.get("name", "Patient"),
            patient_age=patient.get("age", 45),
            patient_gender=patient.get("gender", "Unknown"),
            avatar_url=avatar_url,
            chief_complaint=facts.get("chiefComplaint") or facts.get("chief_complaint", {}).get("value") if isinstance(facts.get("chief_complaint"), dict) else (facts.get("chief_complaint") or case.get("description", "")),
            initial_statement=initial_stmt,
            tags=tags,
            estimated_minutes=estimated_minutes,
            created_at=created_dt
        )

    patient = getattr(case, "patient_profile", None)
    clinical_facts = getattr(case, "clinical_facts", None)
    
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
        avatar_url="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80",
        chief_complaint=clinical_facts.chief_complaint if clinical_facts and clinical_facts.chief_complaint else case.description,
        initial_statement=None,
        tags=[case.specialty, case.difficulty, "OSCE Simulation"],
        estimated_minutes=15,
        created_at=case.created_at
    )

def format_case_detail(case) -> CaseDetail:
    """Formats a database Case into CaseDetail for learner simulation."""
    if isinstance(case, dict):
        patient_data = case.get("patient") or {}
        raw_persona = patient_data.get("persona")
        if isinstance(raw_persona, dict):
            persona_str = ", ".join(f"{v}" for k, v in raw_persona.items() if v)
        else:
            persona_str = str(raw_persona) if raw_persona else None

        initial_stmt = patient_data.get("initial_statement") or patient_data.get("initialStatement") or case.get("initial_statement") or case.get("opening_statement")

        patient = PatientProfileSchema(
            name=patient_data.get("name", "Patient"),
            age=patient_data.get("age", 45),
            gender=patient_data.get("gender", "Unknown"),
            occupation=patient_data.get("occupation"),
            persona=persona_str,
            initial_statement=initial_stmt,
            avatar_url=patient_data.get("avatar_url") or patient_data.get("avatarUrl")
        )

        physical_findings = [
            PhysicalFindingSchema(
                id=pf.get("id", ""),
                system=pf.get("system", "General"),
                finding=pf.get("name") or pf.get("finding", ""),
                value=pf.get("findingDescription") or pf.get("value", ""),
                description=pf.get("findingDescription") or pf.get("description", "")
            )
            for pf in (case.get("physicalFindings") or case.get("physical_findings") or case.get("examination") or [])
        ]

        investigations = [
            InvestigationSchema(
                id=inv.get("id", ""),
                name=inv.get("name", ""),
                category=inv.get("category", "Diagnostics"),
                result=inv.get("result") or inv.get("interpretation", ""),
                unit=inv.get("unit"),
                reference_range=inv.get("reference_range") or inv.get("normalRange"),
                is_available=inv.get("is_available", True)
            )
            for inv in (case.get("investigations") or [])
        ]

        # Candidate diagnosis options for the learner without revealing the correct answer
        raw_dx = case.get("diagnosis_model", {}).get("differentials", []) or case.get("diagnosisOptions", [])
        diagnosis_options = []
        for d in raw_dx:
            if isinstance(d, dict):
                diagnosis_options.append({
                    "id": d.get("id", ""),
                    "name": d.get("name", ""),
                    "category": d.get("category", case.get("specialty", "General"))
                })

        # Candidate management options for the learner without revealing the correct answer
        raw_mgmt = case.get("management_model", {})
        mgmt_list = []
        if isinstance(raw_mgmt, dict):
            for act in raw_mgmt.get("immediate_actions", []):
                mgmt_list.append({"id": f"mgmt-{len(mgmt_list)+1}", "label": act})
            for act in raw_mgmt.get("contraindicated_actions", []):
                mgmt_list.append({"id": f"mgmt-{len(mgmt_list)+1}", "label": act})
        elif isinstance(case.get("managementProtocols"), list):
            for m in case.get("managementProtocols"):
                if isinstance(m, dict):
                    mgmt_list.append({"id": m.get("id", ""), "label": m.get("label", "")})

        avatar_url = patient_data.get("avatar_url") or patient_data.get("avatarUrl") or case.get("avatar_url")
        if not avatar_url:
            avatar_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80" if patient_data.get("gender") == "Male" else "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&auto=format&fit=crop&q=80"

        created = case.get("created_at")
        if isinstance(created, (int, float)):
            created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
        elif isinstance(created, datetime):
            created_dt = created
        elif isinstance(created, str):
            try:
                created_dt = datetime.fromisoformat(created)
            except Exception:
                created_dt = datetime.now(timezone.utc)
        else:
            created_dt = datetime.now(timezone.utc)

        return CaseDetail(
            id=case.get("id", ""),
            title=case.get("title", ""),
            specialty=case.get("specialty", "General"),
            description=case.get("description", ""),
            difficulty=case.get("difficulty", "Intermediate"),
            is_published=case.get("is_published", True),
            patient=patient,
            avatar_url=avatar_url,
            initial_statement=initial_stmt,
            tags=case.get("tags") or [case.get("specialty", "General"), case.get("difficulty", "Intermediate")],
            estimated_minutes=case.get("estimated_minutes") or case.get("estimatedMinutes") or 15,
            triage_nurse_note=case.get("triage_nurse_note") or case.get("triageNurseNote") or case.get("description", ""),
            initial_vitals=case.get("initial_vitals") or case.get("initialVitals") or patient_data.get("initial_vitals"),
            learning_objectives=case.get("learning_objectives") or case.get("learningObjectives") or [
                "Complete targeted clinical history",
                "Perform relevant physical examinations",
                "Formulate evidence-based differential diagnosis"
            ],
            physical_findings=physical_findings,
            investigations=investigations,
            diagnosis_options=diagnosis_options,
            management_protocols=mgmt_list,
            created_at=created_dt
        )

    patient = None
    if getattr(case, "patient_profile", None):
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
        for pf in (getattr(case, "physical_findings", None) or [])
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
        for inv in (getattr(case, "investigations", None) or [])
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
