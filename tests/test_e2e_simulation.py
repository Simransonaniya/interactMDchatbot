"""
InteractMD — End-to-End Simulation Lifecycle Test.
Executes a complete, authentic clinical simulation session from MongoDB:
1. Create session with case_id and case_version.
2. Progressive interview questions and answers.
3. Perform physical exam maneuver.
4. Order diagnostic investigation.
5. Submit differential and primary diagnosis.
6. Submit clinical management.
7. End encounter and receive Attending Physician OSCE evaluation.
"""

import pytest
import uuid
from mongo_db import mongo_manager
from ai_orchestrator import ai_orchestrator
from services.examination_service import examination_service
from services.investigation_service import investigation_service
from services.clinical_submission_service import clinical_submission_service
from services.evaluation_service import evaluation_service


def test_full_end_to_end_clinical_simulation():
    # 1. Start Session
    session_id = str(uuid.uuid4())
    case_id = "chest_pain_001"

    mongo_manager.create_session({
        "session_id": session_id,
        "id": session_id,
        "case_id": case_id,
        "case_version": 1,
        "status": "ACTIVE"
    })

    # 2. Dialogue Sequence
    # Turn 1: Opening
    t1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Hello, what brought you to the emergency room today?",
        session_id=session_id
    )
    assert "pressure" in t1["reply"].lower() or "chest" in t1["reply"].lower()

    # Turn 2: Onset
    t2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="When did this start?",
        session_id=session_id
    )
    assert "45 minutes" in t2["reply"]

    # Turn 3: Location
    t3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Where exactly do you feel it?",
        session_id=session_id
    )
    assert "middle of my chest" in t3["reply"].lower() or "chest" in t3["reply"].lower()

    # Turn 4: Radiation
    t4 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Does it radiate anywhere?",
        session_id=session_id
    )
    assert "jaw" in t4["reply"].lower() or "arm" in t4["reply"].lower()

    # Turn 5: Severity
    t5 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="How severe is it on a scale of 1 to 10?",
        session_id=session_id
    )
    assert "8" in t5["reply"]

    # Turn 6: Associated symptom (SOB)
    t6 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Are you short of breath?",
        session_id=session_id
    )
    assert "yes" in t6["reply"].lower()

    # Turn 7: Negative symptom (Fever)
    t7 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Did you have a fever?",
        session_id=session_id
    )
    assert "no" in t7["reply"].lower()

    # Turn 8: Unknown symptom (Body aches)
    t8 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Did you have body pain or body aches?",
        session_id=session_id
    )
    assert ("haven't really noticed" in t8["reply"].lower() or "not sure" in t8["reply"].lower() or "haven't noticed" in t8["reply"].lower())

    # 3. Perform Physical Examination
    exam_res = examination_service.perform_exam(
        case_id=case_id,
        session_id=session_id,
        system="Cardiovascular"
    )
    assert len(exam_res["findings"]) >= 1

    # 4. Order Diagnostic Investigations
    inv_res = investigation_service.order_investigation(
        case_id=case_id,
        test_id="inv-ecg",
        session_id=session_id
    )
    assert "ST Elevation" in inv_res["result"]["result"]

    # 5. Submit Working Diagnosis
    diag_res = clinical_submission_service.submit_diagnosis(
        session_id=session_id,
        case_id=case_id,
        differential=["Inferior STEMI", "Aortic Dissection"],
        most_likely="Acute ST-Elevation Myocardial Infarction (Inferior STEMI)",
        reasoning="Crushing retrosternal chest pain with ST elevation in II, III, aVF."
    )
    assert diag_res["is_correct_primary"] is True

    # 6. Submit Clinical Management
    mgmt_res = clinical_submission_service.submit_management(
        session_id=session_id,
        case_id=case_id,
        immediate_actions=["Administer chewable Aspirin 324 mg STAT", "Activate Cath Lab for Emergent PCI"],
        treatment=["Heparin IV bolus"],
        escalation="Transfer to catheterization suite"
    )
    assert mgmt_res["status"] == "RECORDED"

    # 7. Complete Simulation & Receive OSCE Evaluation
    eval_res = evaluation_service.evaluate_session(
        case_id=case_id,
        session_id=session_id,
        performed_exam_ids=["exam-cv"],
        ordered_investigation_ids=["inv-ecg"],
        primary_diagnosis_id="dx-stemi",
        differential_diagnosis_ids=["dx-stemi", "dx-dissection"],
        selected_management_ids=["mgmt-aspirin", "mgmt-cath"],
        clinical_rationale="Classic inferior STEMI presentation.",
        duration_seconds=240
    )

    assert eval_res.overall_score >= 70
    assert eval_res.pass_status is True

    # Verify session in MongoDB is now COMPLETED
    final_session = mongo_manager.get_session(session_id)
    assert final_session["status"] == "COMPLETED"

    # Verify messages persisted
    saved_messages = mongo_manager.get_session_messages(session_id)
    assert len(saved_messages) >= 16
