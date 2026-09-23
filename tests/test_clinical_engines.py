"""
InteractMD — Clinical Simulation Engines Tests.
Verifies:
1. Physical Examination Engine (supported vs unsupported systems).
2. Diagnostic Investigation Engine (ECG, Troponin, Peak Flow, Ultrasound, CBC).
3. Diagnosis Submission Engine (differential recording & ground truth comparison).
4. Management Submission Engine (immediate actions & treatment protocols).
5. OSCE Attending Evaluation Engine (5 scoring dimensions).
"""

import pytest
import uuid
from services.examination_service import examination_service
from services.investigation_service import investigation_service
from services.clinical_submission_service import clinical_submission_service
from services.evaluation_service import evaluation_service


def test_01_examination_engine_supported_and_unsupported():
    """Verify examination findings are retrieved from MongoDB and not hallucinated."""
    session_id = str(uuid.uuid4())

    # Supported CV exam
    res_cv = examination_service.perform_exam(
        case_id="chest_pain_001",
        session_id=session_id,
        system="Cardiovascular"
    )
    assert len(res_cv["findings"]) >= 1
    assert "S4 gallop" in res_cv["findings"][0]["findingDescription"]

    # Unsupported / normal system exam
    res_neuro = examination_service.perform_exam(
        case_id="chest_pain_001",
        session_id=session_id,
        system="Neurological"
    )
    assert len(res_neuro["findings"]) >= 1
    assert "Within normal limits" in res_neuro["findings"][0]["findingDescription"]


def test_02_investigation_engine_controlled_results():
    """Verify investigation results come directly from MongoDB."""
    session_id = str(uuid.uuid4())

    # STAT ECG
    res_ecg = investigation_service.order_investigation(
        case_id="chest_pain_001",
        test_id="inv-ecg",
        session_id=session_id
    )
    assert res_ecg["result"]["is_available"] is True
    assert "ST Elevation" in res_ecg["result"]["result"]

    # Peak Flow for Asthma case
    res_pef = investigation_service.order_investigation(
        case_id="dyspnea_002",
        test_id="inv-peak-flow",
        session_id=session_id
    )
    assert "190 L/min" in res_pef["result"]["result"]

    # Ultrasound Appendix for Abdominal case
    res_us = investigation_service.order_investigation(
        case_id="abdominal_pain_003",
        test_id="inv-us-appendix",
        session_id=session_id
    )
    assert "8.2 mm" in res_us["result"]["result"]


def test_03_diagnosis_submission():
    """Verify diagnosis recording and evaluation against case diagnosis model."""
    session_id = str(uuid.uuid4())

    res = clinical_submission_service.submit_diagnosis(
        session_id=session_id,
        case_id="chest_pain_001",
        differential=["ST-Elevation Myocardial Infarction", "Aortic Dissection", "Pulmonary Embolism"],
        most_likely="Acute ST-Elevation Myocardial Infarction (Inferior STEMI)",
        reasoning="Inferior ST elevation with reciprocal changes and crushing chest pain."
    )
    assert res["status"] == "RECORDED"
    assert res["is_correct_primary"] is True


def test_04_management_submission():
    """Verify management protocol recording."""
    session_id = str(uuid.uuid4())

    res = clinical_submission_service.submit_management(
        session_id=session_id,
        case_id="chest_pain_001",
        immediate_actions=["Administer Aspirin 324 mg", "Activate Cath Lab"],
        treatment=["Heparin IV", "Nitroglycerin SL"],
        escalation="Transfer to emergent PCI"
    )
    assert res["status"] == "RECORDED"
    assert len(res["immediate_actions"]) == 2


def test_05_evaluation_engine_scoring_dimensions():
    """Verify 5 OSCE dimensions evaluation."""
    session_id = str(uuid.uuid4())

    eval_res = evaluation_service.evaluate_session(
        case_id="chest_pain_001",
        session_id=session_id,
        conversation_history=[
            {"sender": "LEARNER", "text": "When did the pain start?"},
            {"sender": "LEARNER", "text": "Does it radiate to your jaw or arm?"},
            {"sender": "LEARNER", "text": "Are you having shortness of breath or cold sweats?"},
            {"sender": "LEARNER", "text": "What is your past medical history?"},
            {"sender": "LEARNER", "text": "I'm so sorry, we'll take good care of you."}
        ],
        performed_exam_ids=["exam-cv", "exam-pulm"],
        ordered_investigation_ids=["inv-ecg", "inv-troponin"],
        primary_diagnosis_id="dx-stemi",
        differential_diagnosis_ids=["dx-stemi", "dx-dissection"],
        selected_management_ids=["mgmt-aspirin", "mgmt-cath"],
        clinical_rationale="Inferior STEMI confirmed on ECG.",
        duration_seconds=300
    )

    assert eval_res.overall_score >= 75
    assert eval_res.pass_status is True
    assert len(eval_res.dimensions) == 5

    dim_names = [d.dimension for d in eval_res.dimensions]
    assert "Interview Completeness" in dim_names
    assert "Clinical Reasoning" in dim_names
    assert "Communication" in dim_names
    assert "Empathy" in dim_names
    assert "Management" in dim_names
