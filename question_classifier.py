"""
InteractMD — Question & Clinical Intent Classifier.
Performs semantic intent classification covering:
- HISTORY_QUESTION (OPQRST dimensions, associated symptoms, pertinent negatives, PMH, meds, allergies, family/social history)
- CLARIFICATION ("Are you sure?", "Really?", "Can you explain that again?")
- CONFIRMATION ("Is that correct?", "So it's been going on for 45 minutes?")
- EMPATHY_REASSURANCE ("Take a slow breath", "I'm here with you")
- MANAGEMENT_STATEMENT ("You should rest", "Let's have you sit down", "We'll monitor you")
- MEDICATION_STATEMENT ("Take this medication", "You should take tablet", "Take paracetamol")
- MEDICATION_NAME_FRAGMENT ("niciplus", "paracetomol tablet", "amlodipine")
- MEDICATION_ADHERENCE ("Did you take your medicine today?", "Have you missed any doses?")
- DIAGNOSIS_STATEMENT ("I think this is a heart attack", "This appears to be cardiac")
- EXAM_REQUEST ("I'm going to examine your heart", "Let me listen to your chest")
- INVESTIGATION_REQUEST ("Let's order an ECG", "We should check troponin")
- OFF_TOPIC ("hairfall", "favorite movie", "weather")
- UNKNOWN / UNCLEAR ("how was it?", single characters)
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

from medical_nlu.normalizer import MedicalNormalizer, NormalizedText
from medical_nlu.medication_lexicon import MedicationLexicon
from medical_nlu.entity_extractor import MedicalEntityExtractor, MedicalEntity
from medical_nlu.intent_classifier import MedicalIntentClassifier, MedicalIntent, StructuredNLUResult


class IntentCategory(str, Enum):
    # Standard Core Intent Categories
    HISTORY_QUESTION = "HISTORY_QUESTION"
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    CHALLENGE = "CHALLENGE"
    EMPATHY_REASSURANCE = "EMPATHY_REASSURANCE"
    MANAGEMENT_STATEMENT = "MANAGEMENT_STATEMENT"
    MEDICATION_STATEMENT = "MEDICATION_STATEMENT"
    MEDICATION_NAME_FRAGMENT = "MEDICATION_NAME_FRAGMENT"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
    MEDICATION_HISTORY = "MEDICATION_HISTORY"
    DIET_HISTORY = "DIET_HISTORY"
    DIAGNOSIS_STATEMENT = "DIAGNOSIS_STATEMENT"
    EXAM_REQUEST = "EXAM_REQUEST"
    INVESTIGATION_REQUEST = "INVESTIGATION_REQUEST"
    OFF_TOPIC = "OFF_TOPIC"
    UNKNOWN = "UNKNOWN"

    # Specific Sub-Dimensions & Aliases
    GREETING = "GREETING"
    OPENING_COMPLAINT = "OPENING_COMPLAINT"
    ONSET_TIMING = "ONSET_TIMING"
    ONSET_ACTIVITY = "ONSET_ACTIVITY"
    TIMING = "TIMING"
    LOCATION = "LOCATION"
    CHARACTER = "CHARACTER"
    SEVERITY = "SEVERITY"
    RADIATION = "RADIATION"
    AGGRAVATING_FACTORS = "AGGRAVATING_FACTORS"
    RELIEVING_FACTORS = "RELIEVING_FACTORS"
    ASSOCIATED_SYMPTOM = "ASSOCIATED_SYMPTOM"
    PAST_MEDICAL_HISTORY = "PAST_MEDICAL_HISTORY"
    MEDICATIONS = "MEDICATIONS"
    ALLERGIES = "ALLERGIES"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    SOCIAL_HISTORY = "SOCIAL_HISTORY"
    GENDER_INAPPLICABLE = "GENDER_INAPPLICABLE"
    EMPATHY = "EMPATHY"
    EXAMINATION_REQUEST = "EXAMINATION_REQUEST"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
    SMALL_TALK = "SMALL_TALK"
    UNCLEAR = "UNCLEAR"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class ClassifiedIntent:
    def __init__(
        self,
        raw_query: str,
        category: IntentCategory,
        subconcept: Optional[str] = None,
        slots: Optional[List[str]] = None,
        time_reference: Optional[str] = None,
        ui_category: str = "General",
        empathy_detected: bool = False,
        jargon_detected: Optional[str] = None,
        action_target: Optional[str] = None,
        treatment_substance: Optional[str] = None,
        normalized_query: Optional[str] = None,
        is_follow_up_to_previous_turn: bool = False,
        relationship: Optional[str] = None,
        referenced_topic: Optional[str] = None,
        referenced_slot: Optional[str] = None,
        entities: Optional[List[Any]] = None,
        confidence: float = 1.0,
        topic: Optional[str] = None,
        slot: Optional[str] = None
    ):
        self.raw_query = raw_query
        self.category = category
        self.subconcept = subconcept
        self.slots = slots or []
        self.time_reference = time_reference or "unspecified"
        self.ui_category = ui_category
        self.empathy_detected = empathy_detected
        self.jargon_detected = jargon_detected
        self.action_target = action_target
        self.treatment_substance = treatment_substance
        self.normalized_query = normalized_query or raw_query
        self.is_follow_up_to_previous_turn = is_follow_up_to_previous_turn
        self.relationship = relationship or ("PREVIOUS_TURN" if is_follow_up_to_previous_turn else "NEW_QUESTION")
        self.referenced_topic = referenced_topic
        self.referenced_slot = referenced_slot
        self.entities = entities or []
        self.confidence = confidence
        self.topic = topic or subconcept or category.value
        self.slot = slot or (self.slots[0] if self.slots else self.subconcept)

    @property
    def primary_type(self) -> str:
        """Returns the primary standardized intent category name."""
        if self.category in [
            IntentCategory.CHARACTER, IntentCategory.RADIATION, IntentCategory.SEVERITY,
            IntentCategory.LOCATION, IntentCategory.ONSET_TIMING, IntentCategory.ONSET_ACTIVITY,
            IntentCategory.TIMING, IntentCategory.AGGRAVATING_FACTORS, IntentCategory.RELIEVING_FACTORS,
            IntentCategory.ASSOCIATED_SYMPTOM, IntentCategory.PAST_MEDICAL_HISTORY,
            IntentCategory.MEDICATIONS, IntentCategory.MEDICATION_HISTORY, IntentCategory.ALLERGIES,
            IntentCategory.FAMILY_HISTORY, IntentCategory.SOCIAL_HISTORY, IntentCategory.DIET_HISTORY,
            IntentCategory.OPENING_COMPLAINT, IntentCategory.GENDER_INAPPLICABLE
        ]:
            return "HISTORY_QUESTION"
        if self.category == IntentCategory.MEDICATION_ADHERENCE:
            return "HISTORY_QUESTION"
        if self.category in [IntentCategory.EMPATHY, IntentCategory.EMPATHY_REASSURANCE]:
            return "EMPATHY_REASSURANCE"
        if self.category in [IntentCategory.EXAMINATION_REQUEST, IntentCategory.EXAM_REQUEST]:
            return "EXAM_REQUEST"
        if self.category in [IntentCategory.OUT_OF_SCOPE, IntentCategory.OFF_TOPIC]:
            return "OFF_TOPIC"
        if self.category in [IntentCategory.UNCLEAR, IntentCategory.UNKNOWN]:
            return "UNKNOWN"
        if self.category == IntentCategory.CHALLENGE:
            return "CHALLENGE"
        if self.category == IntentCategory.CLARIFICATION:
            return "CLARIFICATION"
        if self.category == IntentCategory.CONFIRMATION:
            return "CONFIRMATION"
        return self.category.value

    def __repr__(self):
        return f"<ClassifiedIntent category={self.category.value} subconcept={self.subconcept} slots={self.slots} time={self.time_reference} relationship={self.relationship}>"


# Helper normalization functions
def normalize_message(query: str) -> str:
    norm = MedicalNormalizer.normalize(query)
    return norm.normalized_text


def detect_medication_entity(query: str, raw_norm: str) -> Optional[str]:
    lex = MedicationLexicon.get_instance()
    m = lex.match_medication(raw_norm) or lex.match_medication(query)
    if m:
        return m["canonical"]
    return None


def extract_diet_slots_and_time(query: str) -> Tuple[List[str], str]:
    q = query.lower()
    slots = []
    if "breakfast" in q:
        slots.append("breakfast")
    if "lunch" in q:
        slots.append("lunch")
    if "dinner" in q or "supper" in q:
        slots.append("dinner")
    if "snack" in q or "snacks" in q:
        slots.append("snacks")
    if not slots and any(k in q for k in ["eat", "ate", "food", "meal", "diet"]):
        slots.append("general_meal")

    time_ref = "unspecified"
    if any(k in q for k in ["yesterday", "last night", "past day", "previous day", "last evening"]):
        time_ref = "previous_day"
    elif any(k in q for k in ["today", "this morning", "this afternoon", "earlier today", "morning"]):
        time_ref = "today"
    elif any(k in q for k in ["tomorrow"]):
        time_ref = "future"

    return slots, time_ref


# Global singleton instance of MedicalIntentClassifier
_nlu_classifier = MedicalIntentClassifier()


class QuestionClassifier:
    """
    Authoritative Question & Intent Classifier wrapping the Medical Language Understanding Layer.
    """

    @staticmethod
    def classify(query_text: str, session_state: Optional[Any] = None) -> ClassifiedIntent:
        nlu_res: StructuredNLUResult = _nlu_classifier.classify(query_text, session_state=session_state)

        # Map MedicalIntent to legacy IntentCategory and ClassifiedIntent
        intent_map = {
            MedicalIntent.MEDICATION_HISTORY: IntentCategory.MEDICATIONS,
            MedicalIntent.MEDICATION_ADHERENCE: IntentCategory.MEDICATION_ADHERENCE,
            MedicalIntent.MEDICATION_STATEMENT: IntentCategory.MEDICATION_STATEMENT,
            MedicalIntent.MEDICATION_NAME_FRAGMENT: IntentCategory.MEDICATION_NAME_FRAGMENT,
            MedicalIntent.DIET_HISTORY: IntentCategory.SOCIAL_HISTORY,
            MedicalIntent.ONSET_TIMING: IntentCategory.ONSET_TIMING,
            MedicalIntent.ONSET_ACTIVITY: IntentCategory.ONSET_ACTIVITY,
            MedicalIntent.TIMING: IntentCategory.TIMING,
            MedicalIntent.LOCATION: IntentCategory.LOCATION,
            MedicalIntent.CHARACTER: IntentCategory.CHARACTER,
            MedicalIntent.SEVERITY: IntentCategory.SEVERITY,
            MedicalIntent.RADIATION: IntentCategory.RADIATION,
            MedicalIntent.AGGRAVATING_FACTORS: IntentCategory.AGGRAVATING_FACTORS,
            MedicalIntent.RELIEVING_FACTORS: IntentCategory.RELIEVING_FACTORS,
            MedicalIntent.ASSOCIATED_SYMPTOM: IntentCategory.ASSOCIATED_SYMPTOM,
            MedicalIntent.PAST_MEDICAL_HISTORY: IntentCategory.PAST_MEDICAL_HISTORY,
            MedicalIntent.ALLERGIES: IntentCategory.ALLERGIES,
            MedicalIntent.FAMILY_HISTORY: IntentCategory.FAMILY_HISTORY,
            MedicalIntent.SOCIAL_HISTORY: IntentCategory.SOCIAL_HISTORY,
            MedicalIntent.GENDER_INAPPLICABLE: IntentCategory.GENDER_INAPPLICABLE,
            MedicalIntent.CLARIFICATION: IntentCategory.CLARIFICATION,
            MedicalIntent.CONFIRMATION: IntentCategory.CONFIRMATION,
            MedicalIntent.CHALLENGE: IntentCategory.CHALLENGE,
            MedicalIntent.EMPATHY_REASSURANCE: IntentCategory.EMPATHY_REASSURANCE,
            MedicalIntent.MANAGEMENT_STATEMENT: IntentCategory.MANAGEMENT_STATEMENT,
            MedicalIntent.DIAGNOSIS_STATEMENT: IntentCategory.DIAGNOSIS_STATEMENT,
            MedicalIntent.DIAGNOSIS_REQUEST: IntentCategory.DIAGNOSIS_REQUEST,
            MedicalIntent.EXAM_REQUEST: IntentCategory.EXAMINATION_REQUEST,
            MedicalIntent.INVESTIGATION_REQUEST: IntentCategory.INVESTIGATION_REQUEST,
            MedicalIntent.GREETING: IntentCategory.GREETING,
            MedicalIntent.OPENING_COMPLAINT: IntentCategory.OPENING_COMPLAINT,
            MedicalIntent.OFF_TOPIC: IntentCategory.OFF_TOPIC,
            MedicalIntent.PROMPT_INJECTION: IntentCategory.PROMPT_INJECTION,
            MedicalIntent.UNCLEAR: IntentCategory.UNCLEAR,
            MedicalIntent.UNKNOWN: IntentCategory.UNKNOWN,
        }

        category = intent_map.get(nlu_res.intent, IntentCategory.UNKNOWN)

        # Set subconcept and slots
        subconcept = nlu_res.slot or nlu_res.topic
        if nlu_res.intent == MedicalIntent.MEDICATION_HISTORY:
            subconcept = "medications"
            slots = ["current_medications"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_ADHERENCE:
            subconcept = "medication_adherence"
            slots = ["medication_adherence"]
        elif nlu_res.intent == MedicalIntent.DIET_HISTORY:
            subconcept = "diet_history"
            slots = [nlu_res.slot] if nlu_res.slot else ["general_meal"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_STATEMENT:
            subconcept = "medication_instruction"
            slots = ["treatment_order"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_NAME_FRAGMENT:
            subconcept = "medication_fragment"
            slots = ["medication_fragment"]
        elif nlu_res.intent == MedicalIntent.MANAGEMENT_STATEMENT:
            subconcept = "rest_or_monitoring"
            slots = ["rest_or_monitoring"]
        elif nlu_res.intent == MedicalIntent.ONSET_TIMING:
            subconcept = "onset_timing"
            slots = ["onset"]
        elif nlu_res.intent == MedicalIntent.ONSET_ACTIVITY:
            subconcept = "onset_activity"
            slots = ["onset_activity"]
        elif nlu_res.intent == MedicalIntent.ASSOCIATED_SYMPTOM:
            subconcept = nlu_res.slot or "associated_symptoms"
            slots = [nlu_res.slot] if nlu_res.slot else []
        else:
            slots = [nlu_res.slot] if nlu_res.slot else []

        return ClassifiedIntent(
            raw_query=query_text,
            category=category,
            subconcept=subconcept,
            slots=slots,
            time_reference=nlu_res.temporal_reference,
            ui_category=nlu_res.ui_category,
            empathy_detected=nlu_res.empathy_detected,
            action_target=nlu_res.action_target,
            treatment_substance=nlu_res.treatment_substance,
            normalized_query=nlu_res.normalized_query,
            is_follow_up_to_previous_turn=nlu_res.is_follow_up,
            relationship=nlu_res.relationship,
            referenced_topic=nlu_res.referenced_topic,
            referenced_slot=nlu_res.referenced_slot,
            entities=nlu_res.entities,
            confidence=nlu_res.confidence,
            topic=nlu_res.topic,
            slot=nlu_res.slot
        )
