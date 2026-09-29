"""
InteractMD — Question & Clinical Intent Classifier.
Performs semantic intent classification covering:
- HISTORY_QUESTION (OPQRST dimensions, associated symptoms, pertinent negatives, PMH, meds, allergies, family/social history)
- MEDICATION_HISTORY ("What medicines do you take?", "Did you eat medcian?")
- MEDICATION_ADHERENCE ("Did you take your medicine today?", "Have you missed any doses?")
- MEDICATION_STATEMENT ("Take this medication", "You should take tablet", "Take paracetamol")
- MEDICATION_NAME_FRAGMENT ("niciplus", "paracetomol tablet", "amlodipine")
- MEDICATION_NAME_QUERY ("What is amlodipine?", "What is this medicine?")
- MEDICATION_PURPOSE_QUERY ("What is amlodipine for?", "Why are you taking atorvastatin?")
- MEDICATION_DOSAGE_QUERY ("What dose of amlodipine do you take?", "How many milligrams?")
- MEDICATION_FREQUENCY_QUERY ("How often do you take it?", "Do you take it every day?")
- MEDICATION_ROUTE_QUERY ("Is it a tablet?", "Is it an inhaler?", "Do you inject it?")
- MEDICATION_SIDE_EFFECT_QUERY ("Does this medicine make you dizzy?", "Any side effects?")
- MEDICATION_ALLERGY_QUERY ("Are you allergic to any medicine?", "Any drug allergies?")
- MEDICATION_DURATION_QUERY ("How long have you been taking it?")
- CLARIFICATION, CONFIRMATION, CHALLENGE, EMPATHY_REASSURANCE, MANAGEMENT_STATEMENT
- DIAGNOSIS_STATEMENT, EXAM_REQUEST, INVESTIGATION_REQUEST, OFF_TOPIC, UNKNOWN
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

from medical_nlu.normalizer import MedicalNormalizer, NormalizedText
from medical_nlu.medication_catalog import MedicationCatalog
from medical_nlu.entity_extractor import MedicalEntityExtractor, MedicalEntity
from medical_nlu.intent_classifier import MedicalIntentClassifier, MedicalIntent, StructuredNLUResult, ExtractedMedicationSlot


class IntentCategory(str, Enum):
    # Standard Core Intent Categories
    HISTORY_QUESTION = "HISTORY_QUESTION"
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    CHALLENGE = "CHALLENGE"
    EMPATHY_REASSURANCE = "EMPATHY_REASSURANCE"
    MANAGEMENT_INSTRUCTION = "MANAGEMENT_INSTRUCTION"
    MANAGEMENT_STATEMENT = "MANAGEMENT_STATEMENT"
    LIFESTYLE_MANAGEMENT = "LIFESTYLE_MANAGEMENT"
    CLINICAL_CLAIM = "CLINICAL_CLAIM"
    CLINICAL_INTERPRETATION = "CLINICAL_INTERPRETATION"
    CONTEXTUAL_HISTORY_QUESTION = "CONTEXTUAL_HISTORY_QUESTION"
    MEDICATION_STATEMENT = "MEDICATION_STATEMENT"
    MEDICATION_NAME_FRAGMENT = "MEDICATION_NAME_FRAGMENT"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
    MEDICATION_HISTORY = "MEDICATION_HISTORY"
    MEDICATION_NAME_QUERY = "MEDICATION_NAME_QUERY"
    MEDICATION_PURPOSE_QUERY = "MEDICATION_PURPOSE_QUERY"
    MEDICATION_EFFECT_QUERY = "MEDICATION_EFFECT_QUERY"
    MEDICATION_DOSAGE_QUERY = "MEDICATION_DOSAGE_QUERY"
    MEDICATION_FREQUENCY_QUERY = "MEDICATION_FREQUENCY_QUERY"
    MEDICATION_ROUTE_QUERY = "MEDICATION_ROUTE_QUERY"
    MEDICATION_SIDE_EFFECT_QUERY = "MEDICATION_SIDE_EFFECT_QUERY"
    MEDICATION_ALLERGY_QUERY = "MEDICATION_ALLERGY_QUERY"
    MEDICATION_DURATION_QUERY = "MEDICATION_DURATION_QUERY"
    MEDICATION_UNKNOWN = "MEDICATION_UNKNOWN"
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
        extracted_medication: Optional[ExtractedMedicationSlot] = None,
        confidence: float = 1.0,
        topic: Optional[str] = None,
        slot: Optional[str] = None,
        negated: bool = False,
        message_role: Optional[str] = None,
        patient_state_slots_to_retrieve: Optional[List[str]] = None,
        medication_reference: bool = False,
        diet_reference: bool = False,
        meal: Optional[str] = None,
        temporal_relation: Optional[str] = None,
        lifestyle_behaviors: Optional[List[str]] = None,
        claim_type: Optional[str] = None
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
        self.extracted_medication = extracted_medication
        self.confidence = confidence
        self.topic = topic or subconcept or category.value
        self.slot = slot or (self.slots[0] if self.slots else self.subconcept)
        self.negated = negated
        self.message_role = message_role or category.value
        self.patient_state_slots_to_retrieve = patient_state_slots_to_retrieve or []
        self.medication_reference = medication_reference
        self.diet_reference = diet_reference
        self.meal = meal
        self.temporal_relation = temporal_relation
        self.lifestyle_behaviors = lifestyle_behaviors or []
        self.claim_type = claim_type

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
            IntentCategory.OPENING_COMPLAINT, IntentCategory.GENDER_INAPPLICABLE,
            IntentCategory.MEDICATION_ADHERENCE, IntentCategory.MEDICATION_NAME_QUERY,
            IntentCategory.MEDICATION_PURPOSE_QUERY, IntentCategory.MEDICATION_EFFECT_QUERY,
            IntentCategory.MEDICATION_DOSAGE_QUERY, IntentCategory.MEDICATION_FREQUENCY_QUERY,
            IntentCategory.MEDICATION_ROUTE_QUERY, IntentCategory.MEDICATION_SIDE_EFFECT_QUERY,
            IntentCategory.MEDICATION_ALLERGY_QUERY, IntentCategory.MEDICATION_DURATION_QUERY
        ]:
            return "HISTORY_QUESTION"
        if self.category in [IntentCategory.EMPATHY, IntentCategory.EMPATHY_REASSURANCE]:
            return "EMPATHY_REASSURANCE"
        if self.category in [IntentCategory.MANAGEMENT_INSTRUCTION, IntentCategory.MANAGEMENT_STATEMENT]:
            return "MANAGEMENT_INSTRUCTION"
        if self.category == IntentCategory.LIFESTYLE_MANAGEMENT:
            return "LIFESTYLE_MANAGEMENT"
        if self.category in [IntentCategory.CLINICAL_CLAIM, IntentCategory.CLINICAL_INTERPRETATION]:
            return "CLINICAL_CLAIM"
        if self.category == IntentCategory.CONTEXTUAL_HISTORY_QUESTION:
            return "CONTEXTUAL_HISTORY_QUESTION"
        if self.category == IntentCategory.MEDICATION_STATEMENT:
            return "MEDICATION_STATEMENT"
        if self.category in [IntentCategory.EXAMINATION_REQUEST, IntentCategory.EXAM_REQUEST]:
            return "EXAM_REQUEST"
        if self.category in [IntentCategory.OUT_OF_SCOPE, IntentCategory.OFF_TOPIC]:
            return "OFF_TOPIC"
        if self.category in [IntentCategory.UNCLEAR, IntentCategory.UNKNOWN, IntentCategory.MEDICATION_UNKNOWN]:
            return "UNKNOWN"
        if self.category == IntentCategory.CHALLENGE:
            return "CHALLENGE"
        if self.category == IntentCategory.CLARIFICATION:
            return "CLARIFICATION"
        if self.category == IntentCategory.CONFIRMATION:
            return "CONFIRMATION"
        return self.category.value

    def __repr__(self):
        return f"<ClassifiedIntent category={self.category.value} role={self.message_role} subconcept={self.subconcept} slots={self.slots} time={self.time_reference} relationship={self.relationship}>"


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
            MedicalIntent.MEDICATION_NAME_QUERY: IntentCategory.MEDICATION_NAME_QUERY,
            MedicalIntent.MEDICATION_PURPOSE_QUERY: IntentCategory.MEDICATION_PURPOSE_QUERY,
            MedicalIntent.MEDICATION_EFFECT_QUERY: IntentCategory.MEDICATION_EFFECT_QUERY,
            MedicalIntent.MEDICATION_DOSAGE_QUERY: IntentCategory.MEDICATION_DOSAGE_QUERY,
            MedicalIntent.MEDICATION_FREQUENCY_QUERY: IntentCategory.MEDICATION_FREQUENCY_QUERY,
            MedicalIntent.MEDICATION_ROUTE_QUERY: IntentCategory.MEDICATION_ROUTE_QUERY,
            MedicalIntent.MEDICATION_SIDE_EFFECT_QUERY: IntentCategory.MEDICATION_SIDE_EFFECT_QUERY,
            MedicalIntent.MEDICATION_ALLERGY_QUERY: IntentCategory.MEDICATION_ALLERGY_QUERY,
            MedicalIntent.MEDICATION_DURATION_QUERY: IntentCategory.MEDICATION_DURATION_QUERY,
            MedicalIntent.MEDICATION_UNKNOWN: IntentCategory.MEDICATION_UNKNOWN,
            MedicalIntent.MANAGEMENT_INSTRUCTION: IntentCategory.MANAGEMENT_INSTRUCTION,
            MedicalIntent.LIFESTYLE_MANAGEMENT: IntentCategory.LIFESTYLE_MANAGEMENT,
            MedicalIntent.CLINICAL_CLAIM: IntentCategory.CLINICAL_CLAIM,
            MedicalIntent.CLINICAL_INTERPRETATION: IntentCategory.CLINICAL_INTERPRETATION,
            MedicalIntent.CONTEXTUAL_HISTORY_QUESTION: IntentCategory.CONTEXTUAL_HISTORY_QUESTION,
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
        elif nlu_res.intent == MedicalIntent.MEDICATION_PURPOSE_QUERY:
            subconcept = "medication_purpose"
            slots = ["medication_purpose"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_DOSAGE_QUERY:
            subconcept = "medication_dosage"
            slots = ["medication_dosage"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_FREQUENCY_QUERY:
            subconcept = "medication_frequency"
            slots = ["medication_frequency"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_ROUTE_QUERY:
            subconcept = "medication_route_form"
            slots = ["medication_route_form"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_SIDE_EFFECT_QUERY:
            subconcept = "medication_side_effects"
            slots = ["medication_side_effects"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_ALLERGY_QUERY:
            subconcept = "drug_allergies"
            slots = ["drug_allergies"]
        elif nlu_res.intent == MedicalIntent.DIET_HISTORY:
            subconcept = "diet_history"
            slots = [nlu_res.slot] if nlu_res.slot else ["general_meal"]
        elif nlu_res.intent == MedicalIntent.MEDICATION_STATEMENT:
            subconcept = "medication_instruction"
            slots = []
        elif nlu_res.intent == MedicalIntent.MANAGEMENT_INSTRUCTION:
            subconcept = "management_instruction"
            slots = []
        elif nlu_res.intent == MedicalIntent.LIFESTYLE_MANAGEMENT:
            subconcept = "lifestyle_advice"
            slots = []
        elif nlu_res.intent in [MedicalIntent.CLINICAL_CLAIM, MedicalIntent.CLINICAL_INTERPRETATION]:
            subconcept = nlu_res.claim_type or "clinical_claim"
            slots = []
        elif nlu_res.intent == MedicalIntent.CONTEXTUAL_HISTORY_QUESTION:
            subconcept = "contextual_medication_diet"
            slots = ["medication_timing_with_food", "breakfast"]
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
            extracted_medication=nlu_res.extracted_medication,
            confidence=nlu_res.confidence,
            topic=nlu_res.topic,
            slot=nlu_res.slot,
            negated=nlu_res.negated,
            message_role=nlu_res.message_role,
            patient_state_slots_to_retrieve=nlu_res.patient_state_slots_to_retrieve,
            medication_reference=nlu_res.medication_reference,
            diet_reference=nlu_res.diet_reference,
            meal=nlu_res.meal,
            temporal_relation=nlu_res.temporal_relation,
            lifestyle_behaviors=nlu_res.lifestyle_behaviors,
            claim_type=nlu_res.claim_type
        )
