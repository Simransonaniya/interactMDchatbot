"""
InteractMD — Complete Clinical Cases & Database Seeder.
Populates SQLite (interactmd.db) AND MongoDB Atlas with all 4 authentic OSCE benchmark cases,
full normalized clinical facts, physical findings, investigations, and attending evaluation rubrics.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from database import engine, SessionLocal, Base
import models
from models.case_model import (
    Case,
    PatientProfile,
    ClinicalFact,
    PhysicalFinding,
    Investigation,
    CaseHiddenEvaluation
)
from services.auth_service import create_user, get_user_by_email
from mongo_db import mongo_manager


def seed_all():
    print("========================================================")
    print("  InteractMD — Comprehensive Database Seeder (SQLite + MongoDB)")
    print("========================================================")

    # 1. Ensure SQLite schema
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # 2. Seed Default Users
    print("\n1. Seeding Default User Accounts...")
    admin = create_user(
        db,
        email="admin@interactmd.com",
        password="adminpassword123",
        first_name="InteractMD",
        last_name="Administrator",
        role="ADMIN"
    )
    print(f"  [+] Admin User: admin@interactmd.com (ID: {getattr(admin, 'id', 'admin')})")

    learner = create_user(
        db,
        email="learner@interactmd.com",
        password="password123",
        first_name="Simran",
        last_name="Kaur",
        role="LEARNER"
    )
    print(f"  [+] Learner User: learner@interactmd.com (ID: {getattr(learner, 'id', 'learner')})")

    # 3. Load Cases JSON
    data_paths = [
        os.path.join(os.path.dirname(__file__), "data", "cases.json"),
        os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "data", "cases.json")
    ]

    all_cases = []
    for dp in data_paths:
        if os.path.exists(dp):
            try:
                with open(dp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if not any(c.get("title") == item.get("title") for c in all_cases):
                            all_cases.append(item)
            except Exception as e:
                print(f"  [-] Failed loading {dp}: {e}")

    print(f"\n2. Seeding {len(all_cases)} Clinical Cases into SQLite & MongoDB...")

    for case_data in all_cases:
        cid = case_data.get("case_id") or case_data.get("id") or "CS-DEV"
        title = case_data.get("title", "Clinical Case")
        specialty = case_data.get("specialty", "General")
        desc = case_data.get("description", "")
        diff = case_data.get("difficulty", "Intermediate")
        patient_data = case_data.get("patient", {})

        # Save to MongoDB via mongo_manager (which also updates in-memory cache)
        mongo_manager.save_case(case_data)

        # Save to SQLite
        try:
            # Delete old record if exists to refresh
            existing = db.query(Case).filter((Case.id == cid) | (Case.title == title)).first()
            if existing:
                db.delete(existing)
                db.commit()

            db_case = Case(
                id=cid,
                title=title,
                specialty=specialty,
                description=desc,
                difficulty=diff,
                is_published=True
            )
            db.add(db_case)
            db.flush()

            # Patient Profile
            persona_val = patient_data.get("persona")
            if isinstance(persona_val, dict):
                persona_str = f"Personality: {persona_val.get('personality', 'anxious')}, State: {persona_val.get('emotional_state', 'worried')}"
            else:
                persona_str = str(persona_val) if persona_val else "Anxious, in pain"

            db_patient = PatientProfile(
                id=f"pt-{cid}",
                case_id=cid,
                name=patient_data.get("name", "Patient"),
                age=patient_data.get("age", 45),
                gender=patient_data.get("gender") or patient_data.get("sex", "Unknown"),
                occupation=patient_data.get("occupation", "Employed"),
                persona=persona_str
            )
            db.add(db_patient)

            # Clinical Facts
            facts_data = case_data.get("facts", {})
            db_facts = ClinicalFact(
                id=f"cf-{cid}",
                case_id=cid,
                chief_complaint=facts_data.get("chiefComplaint") or (facts_data.get("chief_complaint", {}).get("value") if isinstance(facts_data.get("chief_complaint"), dict) else facts_data.get("chief_complaint")),
                history=facts_data.get("history") or facts_data.get("hpi", {}).get("statement"),
                onset=facts_data.get("onset"),
                timing=facts_data.get("timing"),
                location=facts_data.get("location"),
                character=facts_data.get("character") or facts_data.get("quality"),
                severity=facts_data.get("severity"),
                radiation=facts_data.get("radiation"),
                aggravating_factors=facts_data.get("aggravating_factors") or facts_data.get("provocationPalliative"),
                relieving_factors=facts_data.get("relieving_factors"),
                associated_symptoms=facts_data.get("associatedSymptoms") or facts_data.get("associated_symptoms") or [],
                past_medical_history=facts_data.get("pastMedicalHistory") or facts_data.get("past_medical_history") or [],
                medications=facts_data.get("medications") or [],
                allergies=facts_data.get("allergies") or [],
                facts_json=facts_data
            )
            db.add(db_facts)

            # Physical Findings
            findings = case_data.get("physicalFindings") or case_data.get("physical_findings") or case_data.get("examination") or []
            for i, pf in enumerate(findings):
                raw_pf_id = pf.get("id") or f"pf_{i+1}"
                pf_id = f"{cid}_{raw_pf_id}"
                db.add(PhysicalFinding(
                    id=pf_id,
                    case_id=cid,
                    system=pf.get("system", "General"),
                    finding=pf.get("name") or pf.get("finding", "Physical Finding"),
                    value=pf.get("findingDescription") or pf.get("value", "Normal"),
                    description=pf.get("description") or pf.get("clinicalSignificance", "")
                ))

            # Investigations
            investigations = case_data.get("investigations", [])
            for i, inv in enumerate(investigations):
                raw_inv_id = inv.get("id") or f"inv_{i+1}"
                inv_id = f"{cid}_{raw_inv_id}"
                db.add(Investigation(
                    id=inv_id,
                    case_id=cid,
                    name=inv.get("name", "STAT Investigation"),
                    category=inv.get("category", "Diagnostics"),
                    result=inv.get("result") or inv.get("interpretation") or inv.get("value", "Normal"),
                    unit=inv.get("unit"),
                    reference_range=inv.get("reference_range") or inv.get("normalRange"),
                    is_available=True
                ))

            # Hidden Evaluation
            dx_model = case_data.get("diagnosis_model") or {}
            scoring = case_data.get("scoring_rubric") or case_data.get("scoringRubric") or {}
            db_hidden = CaseHiddenEvaluation(
                id=f"che-{cid}",
                case_id=cid,
                diagnosis=dx_model.get("primary_diagnosis") or case_data.get("diagnosis", title),
                differential_diagnosis=dx_model.get("differentials") or case_data.get("differential_diagnosis") or [],
                management=case_data.get("management") or [],
                learning_objectives=case_data.get("learningObjectives") or case_data.get("learning_objectives") or [],
                scoring_rubric=scoring
            )
            db.add(db_hidden)

            db.commit()
            print(f"  [+] Seeded case '{cid}' - '{title}' into SQLite & MongoDB")
        except Exception as e:
            print(f"  [-] Error saving case '{cid}' to SQLite: {e}")
            db.rollback()

    db.close()
    print("\n========================================================")
    print("  [SUCCESS] All Users and Cases Seeded into Database!")
    print("========================================================")


if __name__ == "__main__":
    seed_all()
