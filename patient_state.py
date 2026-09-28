"""
InteractMD — Structured Patient State & Active Session Context Manager.
Conceptual state tracking:
patient_state
├── demographics
├── chief_complaint
├── symptoms
├── onset
├── timing
├── character
├── severity
├── radiation
├── associated_symptoms
├── past_medical_history
├── medications
├── allergies
├── social_history
├── physical_findings
├── investigations
├── last_patient_message
├── last_clinician_message
├── last_intent
├── last_disclosed_fact_key
└── revealed_facts
"""

import re
from enum import Enum
from typing import Dict, Any, List, Set, Optional


class FactState(str, Enum):
    KNOWN = "KNOWN"
    AVAILABLE = "AVAILABLE"
    AVAILABLE_NEGATIVE = "AVAILABLE_NEGATIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    NOT_YET_REVEALED = "NOT_YET_REVEALED"
    REVEALED = "REVEALED"


class PatientSimulationState:
    """
    Maintains complete structured clinical state for an active patient simulation.
    Ensures previously revealed information is authoritative and never forgotten or contradicted.
    """

    def __init__(self, case_id: str, session_id: Optional[str] = None):
        self.case_id = case_id
        self.session_id = session_id or "default_session"
        self.demographics: Dict[str, Any] = {}
        self.chief_complaint: str = ""
        self.symptoms: Dict[str, Any] = {}
        self.onset: str = ""
        self.onset_activity: str = ""
        self.timing: str = ""
        self.character: str = ""
        self.severity: str = ""
        self.radiation: str = ""
        self.aggravating_factors: str = ""
        self.relieving_factors: str = ""
        self.associated_symptoms: Dict[str, bool] = {}
        self.past_medical_history: List[str] = []
        self.medications: List[str] = []
        self.allergies: List[str] = []
        self.social_history: Dict[str, Any] = {}
        self.family_history: str = ""
        self.physical_findings: Dict[str, Any] = {}
        self.investigations: Dict[str, Any] = {}
        
        # Maps fact key/id -> authoritative statement previously stated by patient
        self.revealed_facts: Dict[str, str] = {}
        self.revealed_fact_ids: Set[str] = set()
        self.conversation_turns: List[Dict[str, Any]] = []
        self.emotional_state: str = "anxious, in discomfort"

        # Explicit turn tracking
        self.last_patient_message: str = ""
        self.last_clinician_message: str = ""
        self.last_intent: Optional[str] = None
        self.last_question_topic: Optional[str] = None
        self.last_disclosed_fact_key: Optional[str] = None
        self.last_disclosed_statement: Optional[str] = None

    @classmethod
    def from_case(cls, case_data: Dict[str, Any], session_id: Optional[str] = None) -> "PatientSimulationState":
        case_id = case_data.get("case_id") or case_data.get("id") or "clinical_case"
        state = cls(case_id=case_id, session_id=session_id)

        # 1. Demographics & Profile
        patient = case_data.get("patient", {}) if isinstance(case_data, dict) else {}
        gender = patient.get("gender") or patient.get("sex") or case_data.get("patient_gender") or "Unknown"
        init_stmt = (
            patient.get("opening_statement")
            or patient.get("initial_statement")
            or case_data.get("opening_statement")
            or case_data.get("initial_statement")
            or ""
        )
        state.demographics = {
            "name": patient.get("name") or case_data.get("patient_name") or "Patient",
            "age": patient.get("age") or case_data.get("patient_age") or 45,
            "gender": gender,
            "occupation": patient.get("occupation") or case_data.get("patient_occupation") or "Unknown",
            "personality": patient.get("persona", {}).get("personality", "anxious") if isinstance(patient.get("persona"), dict) else "anxious",
            "emotional_state": patient.get("persona", {}).get("emotional_state", "worried") if isinstance(patient.get("persona"), dict) else "worried",
            "initial_statement": init_stmt
        }

        if init_stmt:
            state.last_patient_message = init_stmt

        # 2. History & OPQRST Facts
        history = case_data.get("history", {}) if isinstance(case_data.get("history"), dict) else {}
        facts = case_data.get("facts", {}) if isinstance(case_data.get("facts"), dict) else {}

        def _extract_val(field: Any, fallback: str = "") -> str:
            if isinstance(field, dict):
                return str(field.get("value") or fallback)
            if field:
                return str(field)
            return fallback

        # Chief Complaint
        state.chief_complaint = (
            _extract_val(history.get("chief_complaint"))
            or str(patient.get("presentation_complaint") or "")
            or _extract_val(facts.get("chiefComplaint"))
            or str(case_data.get("chief_complaint") or "")
        )

        # Onset & Timing
        state.onset = (
            _extract_val(history.get("onset_timing"))
            or _extract_val(history.get("onset"))
            or _extract_val(facts.get("onset"))
            or str(case_data.get("onset") or "")
        )
        state.onset_activity = (
            _extract_val(history.get("onset_activity"))
            or str(case_data.get("onset_activity") or "")
        )
        state.timing = (
            _extract_val(history.get("timing"))
            or _extract_val(facts.get("timing"))
            or str(case_data.get("timing") or "")
        )

        # Character / Quality
        state.character = (
            _extract_val(history.get("character"))
            or _extract_val(facts.get("quality"))
            or str(case_data.get("character") or "")
        )

        # Severity
        state.severity = (
            _extract_val(history.get("severity"))
            or _extract_val(facts.get("severity"))
            or str(case_data.get("severity") or "")
        )

        # Radiation
        state.radiation = (
            _extract_val(history.get("radiation"))
            or _extract_val(facts.get("radiation"))
            or str(case_data.get("radiation") or "")
        )

        # Aggravating & Relieving
        state.aggravating_factors = (
            _extract_val(history.get("aggravating_factors"))
            or _extract_val(facts.get("aggravatingFactors"))
            or str(case_data.get("aggravating_factors") or "")
        )
        state.relieving_factors = (
            _extract_val(history.get("relieving_factors"))
            or _extract_val(facts.get("relievingFactors"))
            or _extract_val(facts.get("provocationPalliative"))
            or str(case_data.get("relieving_factors") or "")
        )

        # Associated Symptoms
        assoc = history.get("associated_symptoms", {})
        if isinstance(assoc, dict):
            for k, v in assoc.items():
                if isinstance(v, dict):
                    state.associated_symptoms[k] = bool(v.get("value") is True)
                elif isinstance(v, bool):
                    state.associated_symptoms[k] = v
        elif isinstance(assoc, list):
            for sym in assoc:
                state.associated_symptoms[str(sym).lower().replace(" ", "_")] = True

        # Check pertinent negatives
        pertinent_neg = history.get("pertinent_negatives") or facts.get("pertinentNegatives") or []
        if isinstance(pertinent_neg, list):
            for neg in pertinent_neg:
                key = str(neg).lower().replace(" ", "_")
                if key not in state.associated_symptoms:
                    state.associated_symptoms[key] = False

        # Past Medical History
        pmh = case_data.get("past_medical_history") or history.get("past_medical_history") or facts.get("pastMedicalHistory") or []
        if isinstance(pmh, list):
            state.past_medical_history = [str(x) for x in pmh]
        elif isinstance(pmh, str) and pmh.strip():
            state.past_medical_history = [pmh.strip()]

        # Medications
        meds = case_data.get("medications") or facts.get("medications") or []
        if isinstance(meds, list):
            state.medications = [str(x) for x in meds]
        elif isinstance(meds, str) and meds.strip():
            state.medications = [meds.strip()]

        # Allergies
        allergies = case_data.get("allergies") or facts.get("allergies") or []
        if isinstance(allergies, list):
            state.allergies = [str(x) for x in allergies]
        elif isinstance(allergies, str) and allergies.strip():
            state.allergies = [allergies.strip()]

        # Social History
        soc = case_data.get("social_history") or facts.get("socialHistory") or ""
        state.social_history = {"summary": soc}

        # Family History
        fam = case_data.get("family_history") or facts.get("familyHistory") or ""
        state.family_history = fam if isinstance(fam, str) else " ".join(fam)

        # 3. Mark Opening Statement Facts as AUTHORITATIVELY REVEALED
        if init_stmt:
            state.revealed_facts["initial_statement"] = init_stmt
            lower_init = init_stmt.lower()

            # If opening statement mentions elephant / heavy pressure on chest
            if "elephant" in lower_init or "pressure" in lower_init or "crushing" in lower_init or "chest" in lower_init:
                char_stmt = "It feels like a heavy crushing pressure, almost like an elephant is sitting right in the middle of my chest."
                state.record_disclosure("character", char_stmt)
                state.record_disclosure("location", "Right in the middle of my chest.")

            # If opening statement mentions dizzy / cold sweat
            if "dizzy" in lower_init or "dizziness" in lower_init:
                state.associated_symptoms["dizziness"] = True
                state.record_disclosure("associated_symptoms.dizziness", "Yes, I started feeling dizzy and lightheaded on my way in.")

            if "sweat" in lower_init or "cold sweat" in lower_init:
                state.associated_symptoms["sweating"] = True
                state.record_disclosure("associated_symptoms.sweating", "Yes, I broke out in a cold sweat.")

            if "office" in lower_init or "walking" in lower_init or "stairs" in lower_init:
                if state.onset_activity:
                    state.record_disclosure("onset_activity", state.onset_activity)

        return state

    def record_disclosure(self, fact_key: str, fact_statement: str):
        """Record that a fact has been revealed authoritatively."""
        if not fact_key:
            return
        self.revealed_facts[fact_key] = fact_statement
        self.revealed_fact_ids.add(fact_key)
        self.last_disclosed_fact_key = fact_key
        self.last_disclosed_statement = fact_statement

    def is_disclosed(self, fact_key: str) -> bool:
        """Check if a specific fact was already revealed."""
        return fact_key in self.revealed_facts or fact_key in self.revealed_fact_ids

    def get_revealed_statement(self, fact_key: str) -> Optional[str]:
        """Retrieve the authoritative statement previously given for this fact."""
        return self.revealed_facts.get(fact_key)

    def ingest_conversation_history(self, history: List[Dict[str, Any]]):
        """
        Synchronizes session state from the full conversation history.
        Parses previous patient responses to ensure consistency.
        """
        if not history:
            return

        self.conversation_turns = history
        for turn in history:
            sender = str(turn.get("sender") or turn.get("role", "")).lower()
            text = str(turn.get("text") or turn.get("content") or turn.get("message") or "")
            if not text:
                continue

            if sender in ["learner", "user", "doctor", "clinician"]:
                self.last_clinician_message = text
            elif sender in ["patient", "assistant"]:
                self.last_patient_message = text
                lower_text = text.lower()
                # Track revealed character
                if "elephant" in lower_text or "heavy" in lower_text or "squeezing" in lower_text or "crushing" in lower_text:
                    if "dizzy" in lower_text or "office" in lower_text:
                        self.record_disclosure("character", "It feels like a heavy crushing pressure, almost like an elephant is sitting right in the middle of my chest.")
                    else:
                        self.record_disclosure("character", text)
                # Track revealed radiation
                if "jaw" in lower_text or "arm" in lower_text or "shoulder" in lower_text or "back" in lower_text:
                    if "radiat" in lower_text or "spread" in lower_text or "shoot" in lower_text or "goes into" in lower_text:
                        self.record_disclosure("radiation", text)
                # Track revealed onset
                if "45 minutes" in lower_text or "minutes ago" in lower_text or "hour ago" in lower_text or "hours ago" in lower_text or "started about" in lower_text or "began about" in lower_text:
                    self.record_disclosure("onset_timing", text)
                # Track severity
                if "8 out of 10" in lower_text or "8/10" in lower_text or "severe" in lower_text or "about an 8" in lower_text:
                    self.record_disclosure("severity", text)
                # Track medications
                if "lisinopril" in lower_text or "atorvastatin" in lower_text or "aspirin" in lower_text or "inhaler" in lower_text:
                    self.record_disclosure("medications", text)
                # Track allergies
                if "allerg" in lower_text or "nkda" in lower_text:
                    self.record_disclosure("allergies", text)


class PatientStateManager:
    """In-memory and MongoDB session state repository."""
    _cache: Dict[str, PatientSimulationState] = {}

    @classmethod
    def get_or_create(
        cls,
        case_id: str,
        session_id: Optional[str] = None,
        case_data: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> PatientSimulationState:
        key = f"{case_id}::{session_id or 'default'}"
        if key not in cls._cache:
            if case_data:
                state = PatientSimulationState.from_case(case_data, session_id=session_id)
            else:
                state = PatientSimulationState(case_id=case_id, session_id=session_id)
            cls._cache[key] = state
        else:
            state = cls._cache[key]

        if conversation_history:
            state.ingest_conversation_history(conversation_history)

        return state

    @classmethod
    def clear(cls, case_id: str, session_id: Optional[str] = None):
        key = f"{case_id}::{session_id or 'default'}"
        cls._cache.pop(key, None)
