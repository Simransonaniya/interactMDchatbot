"""
InteractMD — Diagnosis & Management Submission Service.
Stores and validates learner diagnosis and clinical management plans directly into MongoDB.
Does NOT ask the patient dialogue LLM to decide treatments or self-diagnose.
"""

from typing import Dict, Any, List, Optional
from mongo_db import mongo_manager


class ClinicalSubmissionService:

    @staticmethod
    def submit_diagnosis(
        session_id: str,
        case_id: str,
        differential: List[str],
        most_likely: str,
        reasoning: Optional[str] = ""
    ) -> Dict[str, Any]:
        case_doc = mongo_manager.get_case_by_id(case_id) or {}
        dx_model = case_doc.get("diagnosis_model", {})
        correct_dx = (dx_model.get("primary_diagnosis") or "").lower()

        is_correct = None
        if correct_dx and most_likely:
            is_correct = (correct_dx in most_likely.lower() or most_likely.lower() in correct_dx)

        diag_data = {
            "differential": differential,
            "most_likely": most_likely,
            "reasoning": reasoning,
            "is_correct_primary": is_correct
        }

        mongo_manager.record_diagnosis_submission(session_id, diag_data)
        mongo_manager.save_event(session_id, "DIAGNOSIS_SUBMITTED", diag_data)

        return {
            "session_id": session_id,
            "case_id": case_id,
            "status": "RECORDED",
            "differential": differential,
            "most_likely": most_likely,
            "reasoning": reasoning,
            "is_correct_primary": is_correct
        }

    @staticmethod
    def submit_management(
        session_id: str,
        case_id: str,
        immediate_actions: List[str],
        treatment: List[str],
        escalation: Optional[str] = "",
        investigations: Optional[List[str]] = None,
        selected_protocol_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        mgmt_data = {
            "immediate_actions": immediate_actions,
            "treatment": treatment,
            "escalation": escalation,
            "investigations": investigations or [],
            "selected_protocol_ids": selected_protocol_ids or []
        }

        mongo_manager.record_management_submission(session_id, mgmt_data)
        mongo_manager.save_event(session_id, "MANAGEMENT_SUBMITTED", mgmt_data)

        return {
            "session_id": session_id,
            "case_id": case_id,
            "status": "RECORDED",
            "immediate_actions": immediate_actions,
            "treatment": treatment,
            "escalation": escalation
        }


clinical_submission_service = ClinicalSubmissionService()
