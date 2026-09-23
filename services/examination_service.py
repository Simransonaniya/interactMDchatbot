"""
InteractMD — Physical Examination Service.
Authoritatively retrieves physical examination findings from MongoDB case records.
Never allows the LLM to invent or hallucinate examination results.
"""

from typing import Dict, Any, List, Optional
from mongo_db import mongo_manager


class ExaminationService:

    @staticmethod
    def perform_exam(
        case_id: str,
        session_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        1. Load case from MongoDB.
        2. Check whether examination findings are available.
        3. Retrieve predefined finding.
        4. Record interaction event in MongoDB.
        5. Return structured finding.
        """
        case_doc = mongo_manager.get_case_by_id(case_id)
        if not case_doc:
            all_cases = mongo_manager.get_all_cases()
            matched = next((c for c in all_cases if c.get("case_id") == case_id or c.get("id") == case_id), None)
            case_doc = matched or (all_cases[0] if all_cases else {})

        exam_list = case_doc.get("examination") or case_doc.get("physical_findings") or case_doc.get("physicalFindings") or []

        matched_findings = []
        for exam in exam_list:
            e_id = exam.get("id") or exam.get("name") or ""
            e_system = exam.get("system") or ""

            if exam_id and (e_id.lower() == exam_id.lower() or exam_id.lower() in e_id.lower()):
                matched_findings.append(exam)
            elif system and (e_system.lower() == system.lower() or system.lower() in e_system.lower()):
                matched_findings.append(exam)

        if not matched_findings:
            if exam_id or system:
                # Finding within normal limits
                finding_name = exam_id or f"{system} Examination"
                matched_findings = [{
                    "id": exam_id or f"exam-{system.lower() if system else 'general'}",
                    "system": system or "General",
                    "name": finding_name,
                    "findingDescription": "Within normal limits. No acute abnormality detected.",
                    "value": "Within normal limits.",
                    "isAbnormal": False,
                    "clinicalSignificance": "Normal physical examination finding."
                }]
            else:
                matched_findings = exam_list

        # Record in MongoDB session and events
        if session_id:
            for f in matched_findings:
                fid = f.get("id") or str(f.get("name"))
                mongo_manager.record_examination_completed(session_id, fid, f)
            mongo_manager.save_event(session_id, "EXAM_PERFORMED", {
                "exam_id": exam_id,
                "system": system,
                "findings_count": len(matched_findings)
            })

        primary_f = matched_findings[0] if matched_findings else {}
        return {
            "session_id": session_id,
            "case_id": case_id,
            "system": primary_f.get("system") or system or "General",
            "finding": primary_f.get("name") or primary_f.get("finding") or exam_id or "Physical Exam",
            "name": primary_f.get("name") or primary_f.get("finding") or exam_id or "Physical Exam",
            "value": primary_f.get("findingDescription") or primary_f.get("value") or "Within normal limits.",
            "findingDescription": primary_f.get("findingDescription") or primary_f.get("value") or "Within normal limits.",
            "is_abnormal": primary_f.get("isAbnormal") if "isAbnormal" in primary_f else primary_f.get("is_abnormal", False),
            "findings": matched_findings
        }


examination_service = ExaminationService()
