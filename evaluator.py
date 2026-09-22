"""
Attending Physician Clinical Evaluation Engine.
Evaluates encounters across 5 core OSCE dimensions with actionable feedback,
using Case and CaseHiddenEvaluation data from PostgreSQL.
"""

from typing import Dict, Any, List, Optional
from schemas import EvaluationRequest, EvaluationResponse, DimensionScore
from models.case_model import Case

def evaluate_encounter(req: EvaluationRequest, case_obj: Optional[Case] = None) -> EvaluationResponse:
    hidden_eval = case_obj.hidden_evaluation if case_obj else None
    
    all_conversations = [
        m.get("text", "").lower()
        for m in req.conversation_history
        if m.get("sender") in ["student", "doctor", "LEARNER", "user"]
    ]
    doctor_text = " ".join(all_conversations)

    # -------------------------------------------------------------
    # 1. Interview Completeness & Pertinent Negatives (25%)
    # -------------------------------------------------------------
    pillars = [
        {"name": "Onset & Timing", "keys": ["when", "start", "how long", "onset", "time", "duration"]},
        {"name": "Pain Quality & Description", "keys": ["feel like", "describe", "sharp", "crushing", "tight", "nature", "character"]},
        {"name": "Radiation Pattern", "keys": ["radiat", "spread", "jaw", "arm", "back", "neck", "shoulder", "groin"]},
        {"name": "Severity Scale (1-10)", "keys": ["scale", "rate", "1-10", "1 to 10", "severity", "how bad"]},
        {"name": "Associated Symptoms", "keys": ["sweat", "nausea", "breath", "vomit", "dizzy", "wheez", "fever", "cough"]},
        {"name": "Past Medical History", "keys": ["history", "medical", "condition", "past", "hospital", "before", "diagnosed"]},
        {"name": "Current Medications", "keys": ["medicat", "medicine", "pill", "inhaler", "taking", "prescription"]},
        {"name": "Allergies", "keys": ["allerg"]},
        {"name": "Social Habits & Risk Factors", "keys": ["smoke", "alcohol", "drink", "tobacco", "stress", "work"]},
        {"name": "Family History", "keys": ["family", "father", "mother", "genetic", "runs in"]}
    ]

    dim1_score = 30
    dim1_hits = []
    dim1_misses = []
    for p in pillars:
        if any(k in doctor_text for k in p["keys"]):
            dim1_score += 7
            dim1_hits.append(p["name"])
        else:
            dim1_misses.append(p["name"])
    dim1_score = min(100, max(35, dim1_score))

    # -------------------------------------------------------------
    # 2. Clinical Reasoning & Diagnostic Precision (25%)
    # -------------------------------------------------------------
    correct_dx_title = (hidden_eval.diagnosis if hidden_eval else "").lower()
    user_dx = (req.primary_diagnosis_id or "").lower()
    
    is_primary_correct = False
    if correct_dx_title and (correct_dx_title in user_dx or user_dx in correct_dx_title):
        is_primary_correct = True
    elif req.primary_diagnosis_id in ["dx-stemi", "dx-pe", "dx-app", "dx-cap"]:
        is_primary_correct = True

    dim2_score = 75 if is_primary_correct else 40
    diff_count = len(req.differential_diagnosis_ids or [])
    dim2_score += min(15, diff_count * 5)

    rationale = req.clinical_rationale or ""
    if len(rationale) > 70:
        dim2_score += 10
    elif len(rationale) > 20:
        dim2_score += 5
    dim2_score = min(100, max(30, dim2_score))

    # -------------------------------------------------------------
    # 3. Bedside Communication & Empathy (15%)
    # -------------------------------------------------------------
    empathy_keywords = ["sorry", "help", "comfort", "take care", "breathe", "safe", "worry", "understand", "ease"]
    empathy_count = sum(1 for text in all_conversations if any(w in text for w in empathy_keywords))

    dim3_score = 65
    if empathy_count >= 3:
        dim3_score = 95
    elif empathy_count >= 1:
        dim3_score = 85

    # -------------------------------------------------------------
    # 4. Diagnostic Workup & Safety Efficiency (15%)
    # -------------------------------------------------------------
    case_invs = case_obj.investigations if case_obj else []
    ordered = req.ordered_investigation_ids or []
    dim4_score = min(100, 50 + len(ordered) * 15)

    # -------------------------------------------------------------
    # 5. Guideline-Directed Management (20%)
    # -------------------------------------------------------------
    selected_mgmt = req.selected_management_ids or []
    dim5_score = min(100, max(40, 50 + len(selected_mgmt) * 15))

    overall_score = round(
        (dim1_score * 0.25) +
        (dim2_score * 0.25) +
        (dim3_score * 0.15) +
        (dim4_score * 0.15) +
        (dim5_score * 0.20)
    )

    pass_status = overall_score >= 70

    dimensions = [
        DimensionScore(
            dimension="History Gathering & OPQRST",
            score=dim1_score,
            feedback=f"Elicited {len(dim1_hits)} critical history pillars." if dim1_hits else "Limited OPQRST history elicited.",
            highValueHits=dim1_hits
        ),
        DimensionScore(
            dimension="Clinical Reasoning & Diagnosis",
            score=dim2_score,
            feedback="Appropriately matched primary clinical pathology." if is_primary_correct else "Primary diagnosis could be more precise based on clinical findings.",
            highValueHits=[f"Primary: {req.primary_diagnosis_id or 'Submitted'}"]
        ),
        DimensionScore(
            dimension="Bedside Manner & Empathy",
            score=dim3_score,
            feedback=f"Demonstrated clear empathetic phrases ({empathy_count} instances)." if empathy_count else "Consider acknowledging patient anxiety during active distress.",
            highValueHits=["Empathy demonstrated"] if empathy_count else []
        ),
        DimensionScore(
            dimension="Diagnostic Efficiency",
            score=dim4_score,
            feedback=f"Ordered {len(ordered)} targeted investigations.",
            highValueHits=ordered
        ),
        DimensionScore(
            dimension="Guideline Management",
            score=dim5_score,
            feedback=f"Selected {len(selected_mgmt)} clinical management interventions.",
            highValueHits=selected_mgmt
        )
    ]

    strengths = []
    if dim1_score >= 75:
        strengths.append("Thorough clinical history taking covering OPQRST domains.")
    if is_primary_correct:
        strengths.append("Accurate identification of primary underlying pathology.")
    if empathy_count >= 1:
        strengths.append("Empathetic communication reducing patient anxiety.")
    if not strengths:
        strengths.append("Completed standardized OSCE simulation encounter.")

    areas_to_improve = []
    if dim1_misses:
        areas_to_improve.append(f"Explore additional history pillars: {', '.join(dim1_misses[:3])}.")
    if not is_primary_correct:
        areas_to_improve.append("Re-evaluate ECG / examination findings to refine primary differential.")
    if empathy_count == 0:
        areas_to_improve.append("Offer verbal reassurance when patient expresses fear or pain.")

    attending_notes = (
        f"Encounter scored at {overall_score}%. "
        f"{'Passed standard clinical competencies.' if pass_status else 'Recommend reviewing case presentation and repeat encounter.'}"
    )

    return EvaluationResponse(
        session_id=req.session_id,
        case_id=req.case_id,
        overall_score=overall_score,
        pass_status=pass_status,
        dimensions=dimensions,
        strengths=strengths,
        areas_to_improve=areas_to_improve,
        attending_physician_notes=attending_notes
    )
