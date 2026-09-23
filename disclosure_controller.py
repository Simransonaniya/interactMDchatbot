"""
InteractMD — Controlled Disclosure Controller.
Tracks and enforces progressive information disclosure for the clinical session.
Prevents unprompted fact dumping while remembering already disclosed facts.
"""

from typing import Dict, Any, List, Set, Optional
from mongo_db import mongo_manager


class DisclosureController:

    @staticmethod
    def get_revealed_facts(session_id: Optional[str]) -> Set[str]:
        if not session_id:
            return set()
        session = mongo_manager.get_session(session_id)
        if session:
            revealed = session.get("revealed_fact_ids", [])
            return set(revealed)
        return set()

    @staticmethod
    def is_fact_revealed(session_id: Optional[str], fact_id: str) -> bool:
        if not session_id or not fact_id:
            return False
        revealed = DisclosureController.get_revealed_facts(session_id)
        return fact_id in revealed

    @staticmethod
    def record_disclosure(session_id: Optional[str], fact_id: str):
        if not session_id or not fact_id:
            return
        mongo_manager.record_fact_revealed(session_id, fact_id)
