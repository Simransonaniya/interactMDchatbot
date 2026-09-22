"""
Database Seed Script for InteractMD.
MANUAL EXECUTION ONLY:
Run `python seed.py` only if you explicitly want to populate sample development cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database import engine, SessionLocal, Base
import models
from services.auth_service import create_user, get_user_by_email
from models.case_model import (
    Case,
    PatientProfile,
    ClinicalFact,
    PhysicalFinding,
    Investigation,
    CaseHiddenEvaluation
)

def seed():
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. Seed admin and learner accounts
        if not get_user_by_email(db, "admin@interactmd.com"):
            admin = create_user(
                db,
                email="admin@interactmd.com",
                password="adminpassword123",
                first_name="InteractMD",
                last_name="Administrator",
                role="ADMIN"
            )
            print(f"Created admin account: admin@interactmd.com / adminpassword123 (ID: {admin.id})")

        if not get_user_by_email(db, "learner@interactmd.com"):
            learner = create_user(
                db,
                email="learner@interactmd.com",
                password="password123",
                first_name="Simran",
                last_name="Kaur",
                role="LEARNER"
            )
            print(f"Created learner account: learner@interactmd.com / password123 (ID: {learner.id})")

        # 2. Seed development clinical case
        existing_case = db.query(Case).filter(Case.id == "CS-ACS-001").first()
        if not existing_case:
            case_acs = Case(
                id="CS-ACS-001",
                title="Acute Crushing Retrosternal Chest Pain",
                specialty="Cardiology",
                description="58-year-old presenting with acute substernal chest heaviness radiating to the left jaw.",
                difficulty="Intermediate",
                is_published=True
            )
            db.add(case_acs)
            db.flush()

            patient = PatientProfile(
                id="pt-dev-001",
                case_id="CS-ACS-001",
                name="David Miller",
                age=58,
                gender="Male",
                occupation="Architectural Project Manager",
                persona="Anxious, clutching chest, visibly diaphoretic."
            )
            db.add(patient)

            facts = ClinicalFact(
                id="cf-dev-001",
                case_id="CS-ACS-001",
                chief_complaint="Severe crushing pressure in the center of my chest that began 45 minutes ago.",
                history="Started while climbing stairs at the office. Associated with cold sweat and dizziness.",
                onset="about 45 minutes ago while climbing stairs.",
                timing="continuous since it began.",
                location="substernal center of the chest.",
                character="crushing pressure, like an elephant sitting on my chest.",
                severity="8 out of 10",
                radiation="radiates to the left jaw and down the left arm.",
                aggravating_factors="physical exertion",
                relieving_factors="nothing so far",
                associated_symptoms=["Diaphoresis", "Shortness of breath", "Mild nausea", "Dizziness"],
                past_medical_history=["Hypertension for 8 years", "Hyperlipidemia"],
                medications=["Amlodipine 5mg daily", "Atorvastatin 20mg daily"],
                allergies=["No known drug allergies"],
                family_history="Father had a myocardial infarction at age 54.",
                social_history="Smoked 1 pack per day for 20 years, quit 3 years ago. Drinks alcohol socially.",
                smoking="Former smoker (20 pack-years, quit 3 years ago)",
                alcohol="Social wine drinker"
            )
            db.add(facts)

            findings = [
                PhysicalFinding(
                    id="pf-cv-001",
                    case_id="CS-ACS-001",
                    system="Cardiovascular",
                    finding="Precordial Auscultation & Heart Sounds",
                    value="Tachycardic regular rhythm at 98 bpm. S1, S2 present. Soft S4 gallop audible at apex. JVP 3cm.",
                    description="S4 gallop indicates decreased left ventricular compliance during acute ischemia."
                ),
                PhysicalFinding(
                    id="pf-pulm-001",
                    case_id="CS-ACS-001",
                    system="Respiratory",
                    finding="Lung Auscultation",
                    value="Clear bilaterally. No wheezes, crackles, or chest wall tenderness.",
                    description="Rules out pulmonary edema and costochondritis."
                ),
                PhysicalFinding(
                    id="pf-gen-001",
                    case_id="CS-ACS-001",
                    system="General / HEENT",
                    finding="Skin Perfusion & Diaphoresis",
                    value="Cool, clammy skin with forehead diaphoresis. Capillary refill 2.5s.",
                    description="Sympathetic activation secondary to acute myocardial distress."
                )
            ]
            for f in findings:
                db.add(f)

            investigations = [
                Investigation(
                    id="inv-ecg-001",
                    case_id="CS-ACS-001",
                    name="12-Lead Electrocardiogram (STAT)",
                    category="Cardiology / Point-of-Care",
                    result="2.5mm ST-segment elevation in leads II, III, aVF with reciprocal ST depression in leads I, aVL.",
                    unit=None,
                    reference_range="Normal sinus rhythm",
                    is_available=True
                ),
                Investigation(
                    id="inv-trop-001",
                    case_id="CS-ACS-001",
                    name="High-Sensitivity Cardiac Troponin I (STAT)",
                    category="Laboratory / Cardiac Markers",
                    result="380",
                    unit="ng/L",
                    reference_range="< 14 ng/L",
                    is_available=True
                )
            ]
            for inv in investigations:
                db.add(inv)

            hidden_eval = CaseHiddenEvaluation(
                id="che-dev-001",
                case_id="CS-ACS-001",
                diagnosis="Acute Inferior ST-Elevation Myocardial Infarction (STEMI)",
                differential_diagnosis=["Aortic Dissection", "Acute Pulmonary Embolism", "Gastroesophageal Reflux Disease"],
                management=["Aspirin 325mg chewable", "Ticagrelor or Clopidogrel", "Immediate Cardiac Catheterization / PCI Activation", "Heparin anticoagulation"],
                learning_objectives=["Identify STEMI on 12-lead ECG", "Deliver immediate dual antiplatelet therapy", "Activate cardiac catheterization pathway within 90 minutes"],
                scoring_rubric={"ecg_interpretation": 25, "opqrst_history": 25, "management_urgency": 25, "empathy": 25}
            )
            db.add(hidden_eval)

            db.commit()
            print("Seeded development case CS-ACS-001 (David Miller - Acute Chest Pain) into PostgreSQL.")

        print("Database seed finished successfully.")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
