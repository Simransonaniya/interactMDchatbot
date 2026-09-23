"""
InteractMD — Patient Simulation State & Conversation Memory.
Maintains session state, emotional context, and tracks disclosed facts.
"""

from typing import Dict, Any, List, Set, Optional


class PatientSessionState:
    def __init__(self, case_id: str, session_id: Optional[str] = None):
        self.case_id = case_id
        self.session_id = session_id or "default-session"
        self.emotional_state: str = "anxious, in discomfort"
        self.disclosed_facets: Set[str] = set()
        self.ordered_investigations: Set[str] = set()
        self.performed_examinations: Set[str] = set()
        self.history: List[Dict[str, Any]] = []

    def record_disclosure(self, facet: str):
        """Record that a specific clinical facet has been disclosed."""
        if facet:
            self.disclosed_facets.add(facet)

    def is_disclosed(self, facet: str) -> bool:
        """Check if a facet was already revealed."""
        return facet in self.disclosed_facets

    def record_investigation(self, test_id: str):
        self.ordered_investigations.add(test_id.lower())

    def record_examination(self, exam_id: str):
        self.performed_examinations.add(exam_id.lower())

    def add_message(self, sender: str, text: str, category: Optional[str] = None):
        self.history.append({
            "sender": sender,
            "text": text,
            "category": category or "General"
        })

    def get_recent_history(self, max_turns: int = 4) -> List[Dict[str, Any]]:
        return self.history[-max_turns:]


class StateManager:
    """In-memory or persistent store for patient session states."""
    _states: Dict[str, PatientSessionState] = {}

    @classmethod
    def get_state(cls, case_id: str, session_id: Optional[str] = None) -> PatientSessionState:
        key = f"{case_id}::{session_id or 'default'}"
        if key not in cls._states:
            cls._states[key] = PatientSessionState(case_id=case_id, session_id=session_id)
        return cls._states[key]

    @classmethod
    def clear_state(cls, case_id: str, session_id: Optional[str] = None):
        key = f"{case_id}::{session_id or 'default'}"
        if key in cls._states:
            del cls._states[key]
