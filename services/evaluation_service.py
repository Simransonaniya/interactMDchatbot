"""
InteractMD — Attending Physician OSCE Evaluation Service.
Computes objective clinical simulation scores across 5 dimensions:
1. Interview Completeness
2. Clinical Reasoning
3. Communication
4. Empathy
5. Management / Next Steps
Uses MongoDB session history and case scoring rubric.
"""

from typing import Dict, Any, List, Optional
from mongo_db import mongo_manager
from schemas import EvaluationResponse, DimensionScore


class EvaluationService:

    @staticmethod
    def evaluate_session(
        case_id: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        performed_exam_ids: Optional[List[str]] = None,
        ordered_investigation_ids: Optional[List[str]] = None,
        primary_diagnosis_id: str = "",
        differential_diagnosis_ids: Optional[List[str]] = None,
        selected_management_ids: Optional[List[str]] = None,
        clinical_rationale: str = "",
        duration_seconds: int = 0
    ) -> EvaluationResponse:
        case_doc = mongo_manager.get_case_by_id(case_id)
        if not case_doc:
            all_cases = mongo_manager.get_all_cases()
            matched = next((c for c in all_cases if c.get("case_id") == case_id or c.get("id") == case_id), None)
            case_doc = matched or (all_cases[0] if all_cases else {})

        # 1. Pull conversation messages
        messages = []
        if session_id:
            messages = mongo_manager.get_session_messages(session_id)
        if not messages and conversation_history:
            messages = conversation_history

        learner_msgs = [m for m in messages if str(m.get("sender") or m.get("role", "")).lower() in ["learner", "student", "doctor", "user"]]
        learner_text_combined = " ".join([str(m.get("content") or m.get("text") or m.get("message", "")).lower() for m in learner_msgs])

        # 2. Extract rubric and models from MongoDB
        rubric = case_doc.get("scoring_rubric", {})
        high_val_q = rubric.get("high_value_questions", [])
        red_flags = rubric.get("red_flags", [])
        critical_actions = rubric.get("critical_actions", [])
        dx_model = case_doc.get("diagnosis_model", {})
        mgmt_model = case_doc.get("management_model", {})

        # ------------------------------------------------------------------
        # Dimension 1: Interview Completeness (History Taking)
        # ------------------------------------------------------------------
        history_hits = []
        history_points = 0
        total_history_targets = max(len(high_val_q), 4)

        for q in high_val_q:
            q_terms = [t for t in q.lower().split() if len(t) > 3]
            if any(term in learner_text_combined for term in q_terms):
                history_hits.append(q)
                history_points += 1

        # Check core dimensions
        if any(k in learner_text_combined for k in ["when", "how long", "start", "onset"]):
            history_points += 1
        if any(k in learner_text_combined for k in ["where", "radiat", "spread", "jaw", "arm", "point"]):
            history_points += 1
        if any(k in learner_text_combined for k in ["feel like", "describe", "sharp", "crushing", "heavy"]):
            history_points += 1
        if any(k in learner_text_combined for k in ["medicat", "pill", "allerg", "history", "condition"]):
            history_points += 1

        history_score = min(100, max(50, int((history_points / 7.0) * 100)))

        # ------------------------------------------------------------------
        # Dimension 2: Clinical Reasoning (Differential & Investigations)
        # ------------------------------------------------------------------
        reasoning_points = 0
        correct_dx_name = (dx_model.get("primary_diagnosis") or "").lower()
        correct_dx_id = (dx_model.get("primary_diagnosis_id") or "").lower()

        # Check primary diagnosis accuracy
        is_primary_correct = False
        if primary_diagnosis_id:
            p_low = primary_diagnosis_id.lower()
            if (correct_dx_id and correct_dx_id in p_low) or (correct_dx_name and (p_low in correct_dx_name or correct_dx_name in p_low)):
                is_primary_correct = True
                reasoning_points += 40

        # Check diagnostic investigations ordered
        investigations_ordered = ordered_investigation_ids or []
        if session_id:
            sess = mongo_manager.get_session(session_id)
            if sess:
                investigations_ordered = list(set(investigations_ordered + sess.get("ordered_investigations", [])))

        if len(investigations_ordered) >= 1:
            reasoning_points += 30
        if len(investigations_ordered) >= 2:
            reasoning_points += 15

        if len(clinical_rationale.strip()) > 15:
            reasoning_points += 15

        reasoning_score = min(100, max(45, reasoning_points))

        # ------------------------------------------------------------------
        # Dimension 3: Communication (Clarity & Professional Tone)
        # ------------------------------------------------------------------
        comm_score = 85
        jargon_penalty = 0
        for jargon in ["stemi", "diaphoresis", "peritonitis", "bronchospasm", "leukocytosis"]:
            if jargon in learner_text_combined:
                jargon_penalty += 5
        comm_score = max(60, comm_score - jargon_penalty)

        # ------------------------------------------------------------------
        # Dimension 4: Empathy (Bedside Manner)
        # ------------------------------------------------------------------
        empathy_hits = []
        for emp in ["sorry", "help you", "comfortable", "don't worry", "breathe", "take your time", "here for you"]:
            if emp in learner_text_combined:
                empathy_hits.append(emp)

        if len(empathy_hits) >= 2:
            empathy_score = 95
        elif len(empathy_hits) == 1:
            empathy_score = 85
        else:
            empathy_score = 70

        # ------------------------------------------------------------------
        # Dimension 5: Management & Next Steps
        # ------------------------------------------------------------------
        mgmt_points = 0
        selected_mgmt = selected_management_ids or []
        if session_id:
            sess = mongo_manager.get_session(session_id)
            if sess and sess.get("management_submission"):
                m_sub = sess["management_submission"]
                selected_mgmt = list(set(selected_mgmt + m_sub.get("selected_protocol_ids", []) + m_sub.get("immediate_actions", [])))

        if len(selected_mgmt) >= 2:
            mgmt_points += 60
        elif len(selected_mgmt) == 1:
            mgmt_points += 40

        if is_primary_correct:
            mgmt_points += 35

        mgmt_score = min(100, max(50, mgmt_points))

        # ------------------------------------------------------------------
        # Overall Calculation
        # ------------------------------------------------------------------
        overall_score = int(
            (history_score * 0.25) +
            (reasoning_score * 0.25) +
            (comm_score * 0.15) +
            (empathy_score * 0.15) +
            (mgmt_score * 0.20)
        )
        pass_status = overall_score >= 70

        strengths = []
        if history_score >= 80:
            strengths.append("Thorough history taking and OPQRST symptom characterization.")
        if is_primary_correct:
            strengths.append("Accurately identified the primary life-threatening diagnosis.")
        if empathy_score >= 85:
            strengths.append("Demonstrated reassuring, empathetic bedside communication.")
        if not strengths:
            strengths.append("Completed standardized clinical encounter steps.")

        areas_to_improve = []
        if not is_primary_correct:
            areas_to_improve.append(f"Ensure target diagnosis aligns with clinical findings: {dx_model.get('primary_diagnosis', 'Primary Diagnosis')}.")
        if len(investigations_ordered) < 2:
            areas_to_improve.append("Order key diagnostic investigations (e.g. ECG, imaging, or laboratory biomarkers) earlier in the encounter.")
        if empathy_score < 80:
            areas_to_improve.append("Incorporate explicit empathy statements when patients express acute distress.")

        dimensions = [
            DimensionScore(
                dimension="Interview Completeness",
                score=history_score,
                feedback="Coverage of chief complaint, onset, timing, radiation, and pertinent negatives.",
                highValueHits=history_hits
            ),
            DimensionScore(
                dimension="Clinical Reasoning",
                score=reasoning_score,
                feedback="Diagnostic differential formulation and investigation selection.",
                highValueHits=[f"Ordered {len(investigations_ordered)} investigation(s)"]
            ),
            DimensionScore(
                dimension="Communication",
                score=comm_score,
                feedback="Professional phrasing, clarity, and avoidance of overwhelming medical jargon."
            ),
            DimensionScore(
                dimension="Empathy",
                score=empathy_score,
                feedback="Validation of patient anxiety and bedside emotional reassurance.",
                highValueHits=empathy_hits
            ),
            DimensionScore(
                dimension="Management",
                score=mgmt_score,
                feedback="Implementation of evidence-based immediate stabilization protocols."
            )
        ]

        notes = (
            f"Encounter completed in {duration_seconds}s. Overall score: {overall_score}/100. "
            f"{'Student demonstrated proficient clinical competency.' if pass_status else 'Recommend review of guideline management protocols and diagnostic red flags.'}"
        )

        eval_resp = EvaluationResponse(
            session_id=session_id,
            case_id=case_id,
            overall_score=overall_score,
            pass_status=pass_status,
            dimensions=dimensions,
            strengths=strengths,
            areas_to_improve=areas_to_improve,
            attending_physician_notes=notes,
            rubric_breakdown={
                "red_flags": red_flags,
                "critical_actions": critical_actions
            }
        )

        # Save to MongoDB
        if session_id:
            mongo_manager.save_evaluation(eval_resp.model_dump())
            mongo_manager.update_session(session_id, {"status": "COMPLETED"})
            mongo_manager.save_event(session_id, "ENCOUNTER_EVALUATED", {
                "overall_score": overall_score,
                "pass_status": pass_status
            })

        return eval_resp


evaluation_service = EvaluationService()
