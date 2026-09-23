"""
InteractMD — AI Virtual Patient Simulation Engine.
Backed by MongoDB Source of Truth & AI Orchestrator Pipeline.
"""

from typing import Dict, Any, List, Optional
from ai_orchestrator import ai_orchestrator


class AIPatientEngine:

    def __init__(self):
        self.provider_name = ai_orchestrator.provider_name

    def process_turn(
        self,
        case_data: Any,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a doctor/student dialogue turn through the AI Orchestrator pipeline.
        Returns a focused, first-person patient response answering only the question asked.
        """
        if isinstance(case_data, dict):
            case_id = case_data.get("case_id") or case_data.get("id") or "chest_pain_001"
        else:
            case_id = str(case_data or "chest_pain_001")

        result = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message=user_message,
            session_id=session_id,
            conversation_history=conversation_history
        )

        return {
            "reply": result.get("reply", ""),
            "empathy_detected": result.get("empathy_detected", False),
            "category": result.get("category", "General"),
            "provider": result.get("provider", self.provider_name),
            "suggested_topics": result.get("suggested_topics", []),
            "facts_revealed": result.get("facts_revealed", []),
            "session_id": session_id
        }

