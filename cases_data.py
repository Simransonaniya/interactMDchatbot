"""
Clinical Cases Ground Truth Store for InteractMD.
Dynamically loaded from JSON specification in chatbot/data/clinical_cases_store.json.
"""

import os
import json
from typing import Dict, Any, Optional

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "clinical_cases_store.json")

def load_clinical_cases() -> Dict[str, Any]:
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load clinical_cases_store.json: {e}")
    return {}

CLINICAL_CASES: Dict[str, Any] = load_clinical_cases()

def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    return CLINICAL_CASES.get(case_id)
