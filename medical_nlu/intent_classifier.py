"""
InteractMD — Medical Intent Classifier & Clinical Slot Extractor.
Combines semantic normalization, lexical analysis, entity recognition,
verb-object dependency logic, temporal parsing, negation detection, and contextual slot extraction.
"""

import re
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from medical_nlu.normalizer import MedicalNormalizer, NormalizedText
from medical_nlu.entity_extractor import MedicalEntityExtractor, MedicalEntity
from medical_nlu.medication_catalog import MedicationCatalog


class MedicalIntent(str, Enum):
    # Specialized Medication Intents
    MEDICATION_HISTORY = "MEDICATION_HISTORY"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
    MEDICATION_STATEMENT = "MEDICATION_STATEMENT"
    MEDICATION_NAME_FRAGMENT = "MEDICATION_NAME_FRAGMENT"
    MEDICATION_NAME_QUERY = "MEDICATION_NAME_QUERY"
    MEDICATION_EFFECT_QUERY = "MEDICATION_EFFECT_QUERY"
    MEDICATION_PURPOSE_QUERY = "MEDICATION_PURPOSE_QUERY"
    MEDICATION_DOSAGE_QUERY = "MEDICATION_DOSAGE_QUERY"
    MEDICATION_FREQUENCY_QUERY = "MEDICATION_FREQUENCY_QUERY"
    MEDICATION_ROUTE_QUERY = "MEDICATION_ROUTE_QUERY"
    MEDICATION_SIDE_EFFECT_QUERY = "MEDICATION_SIDE_EFFECT_QUERY"
    MEDICATION_ALLERGY_QUERY = "MEDICATION_ALLERGY_QUERY"
    MEDICATION_DURATION_QUERY = "MEDICATION_DURATION_QUERY"
    MEDICATION_UNKNOWN = "MEDICATION_UNKNOWN"

    # Core Clinical Dimensions
    DIET_HISTORY = "DIET_HISTORY"
    SYMPTOM_HISTORY = "SYMPTOM_HISTORY"
    PAST_MEDICAL_HISTORY = "PAST_MEDICAL_HISTORY"
    ALLERGIES = "ALLERGIES"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    SOCIAL_HISTORY = "SOCIAL_HISTORY"
    GENDER_INAPPLICABLE = "GENDER_INAPPLICABLE"

    # Clinical Encounter & Interaction Intents
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    CHALLENGE = "CHALLENGE"
    EMPATHY_REASSURANCE = "EMPATHY_REASSURANCE"
    MANAGEMENT_STATEMENT = "MANAGEMENT_STATEMENT"
    DIAGNOSIS_STATEMENT = "DIAGNOSIS_STATEMENT"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
    EXAM_REQUEST = "EXAM_REQUEST"
    INVESTIGATION_REQUEST = "INVESTIGATION_REQUEST"
    GREETING = "GREETING"
    OPENING_COMPLAINT = "OPENING_COMPLAINT"
    SMALL_TALK = "SMALL_TALK"
    OFF_TOPIC = "OFF_TOPIC"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    UNKNOWN = "UNKNOWN"
    UNCLEAR = "UNCLEAR"

    # Specific OPQRST Sub-Dimensions
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


@dataclass
class ExtractedMedicationSlot:
    raw_name: str
    normalized_name: str
    canonical_name: str
    brand_name: Optional[str] = None
    active_ingredients: List[str] = field(default_factory=list)
    dosage: Optional[str] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    form: Optional[str] = None
    duration: Optional[str] = None
    reason: Optional[str] = None
    adherence: Optional[str] = None
    negated: bool = False
    time_reference: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "canonical_name": self.canonical_name,
            "brand_name": self.brand_name,
            "active_ingredients": self.active_ingredients,
            "dosage": self.dosage,
            "unit": self.unit,
            "frequency": self.frequency,
            "route": self.route,
            "form": self.form,
            "duration": self.duration,
            "reason": self.reason,
            "adherence": self.adherence,
            "negated": self.negated,
            "time_reference": self.time_reference,
            "confidence": self.confidence
        }


@dataclass
class StructuredNLUResult:
    raw_query: str
    normalized_query: str
    intent: MedicalIntent
    primary_type: str
    topic: str
    slot: str
    entities: List[MedicalEntity]
    extracted_medication: Optional[ExtractedMedicationSlot] = None
    temporal_reference: str = "unspecified"
    negated: bool = False
    confidence: float = 1.0
    ui_category: str = "General"
    empathy_detected: bool = False
    action_target: Optional[str] = None
    treatment_substance: Optional[str] = None
    is_follow_up: bool = False
    relationship: str = "NEW_QUESTION"
    referenced_topic: Optional[str] = None
    referenced_slot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value,
            "primary_type": self.primary_type,
            "topic": self.topic,
            "slot": self.slot,
            "entities": [e.to_dict() for e in self.entities],
            "extracted_medication": self.extracted_medication.to_dict() if self.extracted_medication else None,
            "temporal_reference": self.temporal_reference,
            "negated": self.negated,
            "confidence": self.confidence,
            "ui_category": self.ui_category,
            "empathy_detected": self.empathy_detected,
            "action_target": self.action_target,
            "treatment_substance": self.treatment_substance,
            "is_follow_up": self.is_follow_up,
            "relationship": self.relationship,
            "referenced_topic": self.referenced_topic,
            "referenced_slot": self.referenced_slot
        }


class MedicalIntentClassifier:
    """
    Robust medical intent classification engine.
    Ensures accurate verb-object semantic understanding and medication hierarchy.
    """

    def __init__(self):
        self.normalizer = MedicalNormalizer()
        self.entity_extractor = MedicalEntityExtractor()
        self.catalog = MedicationCatalog.get_instance()

    def classify(self, raw_query: str, session_state: Optional[Any] = None) -> StructuredNLUResult:
        # Step 1: Normalization
        norm_result: NormalizedText = self.normalizer.normalize(raw_query)
        norm_text = norm_result.normalized_text
        tokens = norm_result.tokens
        raw_low = raw_query.lower().strip()

        # Step 2: Entity Extraction
        entities: List[MedicalEntity] = self.entity_extractor.extract_entities(norm_text, raw_query)
        med_entities = [e for e in entities if e.type == "MEDICATION"]
        food_entities = [e for e in entities if e.type == "FOOD_MEAL"]
        dose_entities = [e for e in entities if e.type == "DOSAGE"]
        freq_entities = [e for e in entities if e.type == "FREQUENCY"]
        route_entities = [e for e in entities if e.type == "ROUTE"]
        form_entities = [e for e in entities if e.type == "FORM"]
        duration_entities = [e for e in entities if e.type == "DURATION"]
        temporal_entities = [e for e in entities if e.type == "TEMPORAL"]
        negation_entities = [e for e in entities if e.type == "NEGATION"]
        adherence_entities = [e for e in entities if e.type == "ADHERENCE"]
        reason_entities = [e for e in entities if e.type == "REASON"]

        # Temporal & Negation determination
        temporal_val = temporal_entities[0].normalized if temporal_entities else "unspecified"
        is_negated = len(negation_entities) > 0 or any(k in norm_text for k in ["do not", "don't", "dont", "never", "no ", "not taking", "stopped"])

        # Build structured medication slot if medication entity detected
        extracted_med_slot = None
        if med_entities:
            top_med = med_entities[0]
            dose_val = dose_entities[0].metadata.get("dose") if (dose_entities and dose_entities[0].metadata) else None
            unit_val = dose_entities[0].metadata.get("unit") if (dose_entities and dose_entities[0].metadata) else None
            freq_val = freq_entities[0].normalized if freq_entities else None
            route_val = route_entities[0].normalized if route_entities else None
            form_val = form_entities[0].normalized if form_entities else None
            dur_val = duration_entities[0].normalized if duration_entities else None
            adh_val = adherence_entities[0].normalized if adherence_entities else None
            reason_val = reason_entities[0].normalized if reason_entities else None

            # Check if brand
            brand_val = None
            if top_med.metadata and top_med.metadata.get("brand_names"):
                brands = top_med.metadata.get("brand_names")
                if any(b.lower() == top_med.text.lower() for b in brands):
                    brand_val = top_med.text.title()

            extracted_med_slot = ExtractedMedicationSlot(
                raw_name=top_med.text,
                normalized_name=top_med.normalized,
                canonical_name=top_med.normalized,
                brand_name=brand_val,
                active_ingredients=top_med.metadata.get("active_ingredients", [top_med.normalized]) if top_med.metadata else [top_med.normalized],
                dosage=dose_val,
                unit=unit_val,
                frequency=freq_val,
                route=route_val,
                form=form_val,
                duration=dur_val,
                reason=reason_val,
                adherence=adh_val,
                negated=is_negated,
                time_reference=temporal_val if temporal_val != "unspecified" else None,
                confidence=top_med.confidence
            )

        # Extract previous-turn state if available
        last_topic = None
        last_slot = None
        if session_state:
            last_topic = getattr(session_state, "last_topic", None) or getattr(session_state, "last_question_topic", None)
            last_slot = getattr(session_state, "last_slot", None) or getattr(session_state, "last_disclosed_fact_key", None)

        def result(
            intent: MedicalIntent,
            primary_type: str,
            topic: str,
            slot: str,
            ui_category: str = "General",
            confidence: float = 1.0,
            empathy: bool = False,
            action_target: Optional[str] = None,
            treatment_substance: Optional[str] = None,
            is_follow_up: bool = False,
            relationship: str = "NEW_QUESTION",
            temporal: Optional[str] = None
        ) -> StructuredNLUResult:
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=intent,
                primary_type=primary_type,
                topic=topic,
                slot=slot,
                entities=entities,
                extracted_medication=extracted_med_slot,
                temporal_reference=temporal if temporal is not None else temporal_val,
                negated=is_negated,
                confidence=confidence,
                ui_category=ui_category,
                empathy_detected=empathy,
                action_target=action_target,
                treatment_substance=treatment_substance,
                is_follow_up=is_follow_up,
                relationship=relationship,
                referenced_topic=last_topic if is_follow_up else None,
                referenced_slot=last_slot if is_follow_up else None
            )

        # -------------------------------------------------------------
        # 0. UNCLEAR / NOISE SHIELD
        # -------------------------------------------------------------
        if re.fullmatch(r"[^a-zA-Z0-9\s]+", raw_low) or raw_low in ["...", ".", "..", "how was it", "what about that", "and then"]:
            return result(MedicalIntent.UNCLEAR, "UNKNOWN", "general", "unclear", ui_category="General", confidence=0.1)

        # -------------------------------------------------------------
        # 1. SECURITY & PROMPT INJECTION SHIELD
        # -------------------------------------------------------------
        injection_patterns = [
            r"ignore (all )?(previous |your )?instructions",
            r"show (me )?(the )?(complete )?(case )?json",
            r"system prompt",
            r"developer mode",
            r"dump (the )?database",
            r"override rules",
            r"jailbreak",
        ]
        if any(re.search(pat, norm_text) for pat in injection_patterns):
            return result(MedicalIntent.PROMPT_INJECTION, "PROMPT_INJECTION", "security", "injection_defense", ui_category="Security")

        # -------------------------------------------------------------
        # 2. HIDDEN DIAGNOSIS REQUEST & DIAGNOSIS STATEMENTS
        # -------------------------------------------------------------
        diag_request_patterns = [
            r"what is (your|my) diagnosis",
            r"what (condition|disease) do (i|you) have",
            r"tell me (what )?the diagnosis (is)?",
            r"tell me (your |the )?hidden diagnosis",
            r"what do you think is wrong with (me|you)",
            r"what disease do you have",
        ]
        if any(re.search(pat, norm_text) for pat in diag_request_patterns):
            return result(MedicalIntent.DIAGNOSIS_REQUEST, "DIAGNOSIS_REQUEST", "diagnosis", "diagnosis_shield", ui_category="General")

        diag_statement_patterns = [
            r"\b(i think (this is|this may be|this might be|you have|you are having)|this appears to be|this looks like|my impression is|i believe (this is|you have)|could be|may be)\b.*\b(heart attack|cardiac|myocardial|infarction|angina|coronary|anxiety|panic|panic attack|gerd|reflux|acid reflux|costochondritis|muscle strain|pulmonary embolism|pe|asthma|pneumonia)\b",
            r"\b(you (are having|have|might be having)|this is|this may be|it is)\s+(a |an )?(heart attack|cardiac event|cardiac issue|panic attack|angina|stemi|mi)\b",
            r"\b(i suspect|diagnosing you with|working diagnosis is)\b.*\b(cardiac|heart attack|angina|anxiety|gerd)\b",
        ]
        if any(re.search(pat, norm_text) for pat in diag_statement_patterns):
            return result(MedicalIntent.DIAGNOSIS_STATEMENT, "DIAGNOSIS_STATEMENT", "diagnosis", "clinician_diagnosis", ui_category="General")

        # -------------------------------------------------------------
        # 3. EXAMINATION & INVESTIGATION REQUESTS
        # -------------------------------------------------------------
        exam_patterns = [
            r"(i would like to|i want to|let me|can i|i will|let us) (check|take|examine|listen to|perform|do) (your )?(vital signs|vitals|blood pressure|heart rate|pulse|temp|temperature)",
            r"(i would like to|i want to|let me|can i|i will|let us) (perform|do|examine) (a |an )?(cardiovascular|respiratory|chest|abdominal|physical|neuro) (exam|examination)",
            r"(i would like to|i want to|let me|can i|i will|let us) (listen to|auscultate|examine) (your )?(heart|lungs|chest|breathing|abdomen)",
            r"^let us examine (your )?(heart|lungs|chest|breathing|abdomen)",
        ]
        if any(re.search(pat, norm_text) for pat in exam_patterns):
            target = "vitals" if any(v in norm_text for v in ["vital", "blood pressure", "pulse", "temp"]) else "physical_exam"
            return result(MedicalIntent.EXAM_REQUEST, "EXAM_REQUEST", "examination", target, ui_category="Exam", action_target=target)

        investigation_patterns = [
            r"\b(i would like to|i want to|let us|let s|let\'s|can we|i will|order|we should)\s+(to\s+)?(order|run|get|perform|obtain|check)?\s*(a\s+|an\s+)?(stat\s+)?(12-lead\s+)?(ecg|ekg|chest x-ray|cxr|blood test|labs|troponin|ct scan|ultrasound)\b",
            r"\border\s+(a\s+|an\s+)?(stat\s+)?(12-lead\s+)?(ecg|ekg|chest x-ray|cxr|blood test|labs|troponin|ct scan|ultrasound)\b",
            r"what did (my |the )?(stat )?(12-lead )?ecg (show|say)",
            r"what does the ecg show",
            r"what did (my |the )?blood(work| test|s)? (show|say)",
            r"what (is|are) my (troponin|labs|lab results|enzymes)",
        ]
        if any(re.search(pat, norm_text) for pat in investigation_patterns):
            target = "ecg" if ("ecg" in norm_text or "ekg" in norm_text) else "labs"
            return result(MedicalIntent.INVESTIGATION_REQUEST, "INVESTIGATION_REQUEST", "investigations", target, ui_category="Diagnostics", action_target=target)

        # -------------------------------------------------------------
        # 4. DIALOGUE META-INTENTS (Clarification, Confirmation, Challenge)
        # -------------------------------------------------------------
        challenge_patterns = [
            r"\b(how\s+(can\s+you|do\s+you|could\s+you|can|you)\s+(not|do\s+not)\s+know)\b",
            r"\b(how\s+(can\s+you|do\s+you|could\s+you|can|you)\s+(not|do\s+not)\s+remember)\b",
            r"\b(why\s+(cannot\s+you|can\s+you\s+not|do\s+not\s+you|you\s+do\s+not)\s+(know|remember))\b",
            r"\b(are\s+you\s+(made|mad)\s+(that\s+)?(you\s+)?do\s+not\s+know\s+anything)\b",
            r"^how can you do not know\b",
            r"^why you do not know\b",
            r"^you do not know\??$",
            r"^you do not remember\??$",
        ]
        if any(re.search(pat, norm_text) for pat in challenge_patterns):
            return result(MedicalIntent.CHALLENGE, "CHALLENGE", last_topic or "general", last_slot or "challenge", ui_category="General", is_follow_up=True, relationship="CHALLENGE")

        clarification_patterns = [
            r"^(are you sure|really\??|are you certain|you sure|are you positive|are you absolutely sure|are you really sure)\b",
            r"^(can you explain that again|what do you mean|could you clarify|tell me more about that|could you repeat that|can you explain|can you clarify)\b",
            r"^(what do you mean by that|how is that possible|how can you be sure)\b",
            r"^(are you sure about that|you sure about that)\b",
        ]
        if any(re.search(pat, norm_text) for pat in clarification_patterns):
            return result(MedicalIntent.CLARIFICATION, "CLARIFICATION", last_topic or "general", last_slot or "clarification", ui_category="General", is_follow_up=True, relationship="CLARIFICATION")

        confirmation_patterns = [
            r"\b(is that correct|is this correct|am i understanding correctly|am i right)\b",
            r"\b(you said.*(correct|right))\b",
            r"^(so (it is|it has been|it is been|you feel|it started|you have|the pain is))\b",
            r"^so\b.*(minutes|hours|days|weeks|\d+|correct|right|going on)\b",
        ]
        if any(re.search(pat, norm_text) for pat in confirmation_patterns):
            return result(MedicalIntent.CONFIRMATION, "CONFIRMATION", last_topic or "general", last_slot or "confirmation", ui_category="General", is_follow_up=True, relationship="CONFIRMATION")

        # -------------------------------------------------------------
        # 5. EMPATHY & BEDSIDE REASSURANCE
        # -------------------------------------------------------------
        empathy_phrases = [
            "sorry", "concern", "take care", "take good care", "help you", "comfortable",
            "breathe", "stay calm", "don't worry", "take your time", "here for you",
            "make you comfortable", "must be frightening", "understand", "we are going to take care",
            "we are going to take good care", "i hear you", "you are safe", "we will figure this out", "in good hands",
            "take a breath", "take a deep breath", "take a slow breath", "i am here with you",
            "you are going to be okay", "you will be okay", "going through this"
        ]
        empathy_detected = any(p in norm_text for p in empathy_phrases)
        clinical_keywords = [
            "pain", "when did", "where is", "rate", "scale of 1", "sweat", "short of breath", "nausea", "fever",
            "medicine", "medication", "pill", "tablet", "allerg", "vomit", "body ache", "cough", "diarrhea", "inhaler", "start", "condition", "describe", "feel"
        ]
        if empathy_detected and len(tokens) <= 25 and not any(k in norm_text for k in clinical_keywords):
            return result(MedicalIntent.EMPATHY_REASSURANCE, "EMPATHY_REASSURANCE", "general", "empathy", ui_category="General", empathy=True)

        # -------------------------------------------------------------
        # 6. CLINICIAN DIRECTIVES & STATEMENTS (Prescribing / Orders)
        # -------------------------------------------------------------
        med_statement_patterns = [
            r"^(take|try|have|start|administer|give)\s+(a\s+|the\s+|this\s+|some\s+)?(paracetamol|tablet|pill|medicine|medication|aspirin|sertraline|nicip|nicip\s+plus|atorvastatin|amlodipine|inhaler|ibuprofen|tylenol|capsule|drug)\b",
            r"\b(you (should|can|need to|must|have to)|i (will|am going to|can|want to|recommend you)|let us|we (will|should|can))\s+(take|give you|prescribe|administer|try|start you on|take this)\s+(this\s+|some\s+|the\s+|a\s+)?(medicine|medication|pill|drug|tablet|treatment|dose|prescription|paracetamol|aspirin|nicip|nicip\s+plus|atorvastatin|amlodipine|tablet)\b",
            r"\b(you should take|take)\s+(tablet|medicine|a tablet|a pill|paracetamol|aspirin|sertraline|nicip\s+plus|atorvastatin|amlodipine|this tablet|this medication)\b",
            r"\b(prescribe|prescribing|order)\s+(medication|medicine|pill|drug|tablet|treatment)\b",
            r"\b(i am giving you|i will give you|let me give you)\s+(some\s+|a\s+|the\s+)?(medicine|medication|pill|tablet|drug|dose)\b",
            r"^start\s+(this\s+|a\s+|the\s+)?(tablet|medicine|medication|pill|drug)\b",
            r"^you\s+can\s+take\s+the\s+medicine\b"
        ]
        is_interrogative = any(norm_text.startswith(w) for w in ["did you", "do you", "have you", "had you", "are you", "what", "which", "can you tell", "how", "why", "is it", "does", "any"])

        if not is_interrogative and any(re.search(pat, norm_text) for pat in med_statement_patterns):
            substance = med_entities[0].normalized if med_entities else "medication"
            return result(MedicalIntent.MEDICATION_STATEMENT, "MEDICATION_STATEMENT", "medication", "treatment_order", ui_category="Management", empathy=empathy_detected, treatment_substance=substance)

        # Medication Name Fragment
        clean_words = re.sub(r"[^\w\s]", "", norm_text).strip()
        is_standalone_med = False
        if len(tokens) <= 3:
            if med_entities and len(med_entities) > 0:
                med_name = med_entities[0].normalized
                matched = med_entities[0].text
                if clean_words in [matched, f"{matched} tablet", f"{matched} tablets", f"take {matched}", f"{matched} pill", f"{matched} pills", med_name, f"{med_name} tablet", f"{med_name} tablets"]:
                    is_standalone_med = True

        if is_standalone_med and not is_interrogative:
            substance = med_entities[0].normalized if med_entities else "medication"
            if clean_words.startswith("take"):
                return result(MedicalIntent.MEDICATION_STATEMENT, "MEDICATION_STATEMENT", "medication", "treatment_order", ui_category="Management", empathy=empathy_detected, treatment_substance=substance)
            else:
                return result(MedicalIntent.MEDICATION_NAME_FRAGMENT, "MEDICATION_NAME_FRAGMENT", "medication", "medication_fragment", ui_category="Management", empathy=empathy_detected, treatment_substance=substance)

        # Non-pharmacological Management Statement
        mgmt_patterns = [
            r"\b(you (should|can|need to|must|have to)|let us|try to|i want you to|we will|we shall|let us have you|i would like you to)\s+(take (some |a )?rest|rest|sit down|lie down|relax|stay in bed|stay still|take it easy|stay calm|monitor you|keep you under observation)\b",
            r"\b(sit down and rest|have you sit down|have you lie down|sit down|lie down|take a seat|have a seat|rest for a bit|rest now)\b",
            r"^(you should take (some |a )?rest|take (some |a )?rest|have (some |a )?rest|sit down|lie down|rest now|rest a bit|we will monitor you)\b",
            r"\b(we will monitor you|monitor you)\b",
            r"\b(take (some |a )?rest)\b",
        ]
        if any(re.search(pat, norm_text) for pat in mgmt_patterns):
            return result(MedicalIntent.MANAGEMENT_STATEMENT, "MANAGEMENT_STATEMENT", "management", "rest_or_monitoring", ui_category="Management", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 7. SPECIALIZED MEDICATION INTENTS (ALLERGIES, SIDE EFFECTS, PURPOSE, DOSAGE, FREQUENCY, ROUTE, DURATION, ADHERENCE)
        # -------------------------------------------------------------
        # 7A. Medication Allergy Query
        allergy_patterns = [
            r"\b(allergic\s+to\s+(any\s+|the\s+)?(medication|medicine|medicines|drug|drugs|pills?|penicillin))\b",
            r"\b(which\s+(medicines?|medications?|drugs?)\s+are\s+you\s+allergic\s+to)\b",
            r"\b(any\s+(drug\s+|medication\s+|medicine\s+)?allerg(y|ies))\b",
            r"\b(have\s+you\s+had\s+an\s+allergic\s+reaction\s+to\s+(any\s+)?(drug|medicine|medication|penicillin))\b",
            r"^are\s+you\s+allergic\s+to\s+(any\s+)?(medicine|medication|drug|penicillin)\b",
            r"^any\s+drug\s+allergies\b",
            r"^are\s+you\s+allergic\s+to\s+penicillin\b"
        ]
        if any(re.search(pat, norm_text) for pat in allergy_patterns) or (
            any(k in norm_text for k in ["allerg", "allergic"]) and any(k in norm_text for k in ["medicine", "medication", "drug", "penicillin", "which"])
        ):
            return result(MedicalIntent.MEDICATION_ALLERGY_QUERY, "HISTORY_QUESTION", "allergies", "drug_allergies", ui_category="Allergies", empathy=empathy_detected)

        # 7B. Medication Side Effect Query
        side_effect_patterns = [
            r"\b(cause|causing|causes|make\s+you|makes\s+you|give\s+you|gives\s+you|lead\s+to)\s+(dizzy|dizziness|nausea|nauseous|headache|swelling|cough|side\s+effects?|problems?|vomiting)\b",
            r"\b(side\s+effects?|adverse\s+effects?|adverse\s+reactions?|reactions?)\s*(from|of|with)?\s*(this\s+|the\s+|your\s+)?(medicine|medication|tablet|pill|amlodipine|atorvastatin|[a-z]+)?\b",
            r"\b(does\s+(this\s+|the\s+|your\s+)?(medicine|medication|tablet|pill|amlodipine|atorvastatin|[a-z]+)\s+(cause|give\s+you|make\s+you))\b",
            r"\b(having\s+problems?\s+from\s+(the\s+|your\s+)?(medication|medicine|tablet|pill|treatment))\b",
            r"\bdid\s+(the\s+|this\s+|your\s+)?(tablet|pill|medicine|medication)\s+cause\b",
            r"\bproblems?\s+from\s+the\s+medication\b",
            r"^\s*any\s+side\s+effects?\b",
            r"^\s*side\s+effects?\??$",
            r"^\s*does\s+this\s+medicine\s+cause\b",
            r"^\s*did\s+the\s+tablet\s+cause\s+nausea\b",
            r"^\s*are\s+you\s+having\s+problems?\s+from\s+the\s+medication\b"
        ]
        if any(re.search(pat, norm_text) for pat in side_effect_patterns) or (
            "side effect" in norm_text or "side effects" in norm_text
        ):
            return result(MedicalIntent.MEDICATION_SIDE_EFFECT_QUERY, "HISTORY_QUESTION", "medication", "medication_side_effects", ui_category="Meds", empathy=empathy_detected)

        # 7C. Medication Purpose / Indication Query
        purpose_patterns = [
            r"\b(what\s+is\s+(this\s+|the\s+|your\s+)?([a-z]+)\s+for)\b",
            r"\b(why\s+(are\s+you\s+taking|do\s+you\s+take|were\s+you\s+prescribed)\s+([a-z]+))\b",
            r"\b(what\s+does\s+(this\s+|the\s+)?(medicine|medication|tablet|pill|[a-z]+)\s+treat)\b",
            r"\b(what\s+is\s+this\s+(medicine|tablet|pill|medication)\s+for)\b",
            r"^what is amlodipine for\b",
            r"^why are you taking atorvastatin\b",
            r"^why do you take atorvastatin\b"
        ]
        if any(re.search(pat, norm_text) for pat in purpose_patterns):
            return result(MedicalIntent.MEDICATION_PURPOSE_QUERY, "HISTORY_QUESTION", "medication", "medication_purpose", ui_category="Meds", empathy=empathy_detected)

        # 7D. Medication Dosage Query
        dosage_patterns = [
            r"\b(how\s+much\s+([a-z]+)\s+do\s+you\s+take)\b",
            r"\b(what\s+dose\s+(are\s+you\s+on|of\s+([a-z]+)\s+do\s+you\s+take|do\s+you\s+take))\b",
            r"\b(how\s+many\s+milligrams|what\s+strength\s+is\s+the\s+tablet|how\s+many\s+mg)\b",
            r"\b(what\s+dose\s+are\s+you\s+on|what\s+dose\s+do\s+you\s+take)\b",
            r"^what dose of amlodipine do you take\b",
            r"^how much amlodipine do you take\b",
            r"^how many milligrams\b",
            r"^what strength is the tablet\b"
        ]
        if any(re.search(pat, norm_text) for pat in dosage_patterns):
            return result(MedicalIntent.MEDICATION_DOSAGE_QUERY, "HISTORY_QUESTION", "medication", "medication_dosage", ui_category="Meds", empathy=empathy_detected)

        # 7E. Medication Frequency Query
        frequency_patterns = [
            r"\b(how\s+often\s+do\s+you\s+take\s+(it|this|them|your\s+medicine|your\s+medication|[a-z]+))\b",
            r"\b(how\s+many\s+times\s+(a\s+day|per\s+day|a\s+week))\b",
            r"\b(do\s+you\s+take\s+it\s+every\s+day|morning\s+or\s+night|when\s+do\s+you\s+take\s+it)\b",
            r"^how often do you take it\b",
            r"^how many times a day\b",
            r"^do you take it every day\b"
        ]
        if any(re.search(pat, norm_text) for pat in frequency_patterns):
            return result(MedicalIntent.MEDICATION_FREQUENCY_QUERY, "HISTORY_QUESTION", "medication", "medication_frequency", ui_category="Meds", empathy=empathy_detected)

        # 7F. Medication Route / Form Query
        route_patterns = [
            r"\b(is\s+it\s+a\s+(tablet|pill|capsule|inhaler|injection|cream|liquid))\b",
            r"\b(do\s+you\s+inject\s+it|how\s+do\s+you\s+take\s+it|is\s+it\s+oral|is\s+it\s+by\s+mouth|is\s+it\s+an\s+inhaler)\b",
            r"^is it a tablet\??$",
            r"^is it an inhaler\??$",
            r"^do you inject it\??$",
            r"^how do you take it\??$",
            r"^is it oral\??$"
        ]
        if any(re.search(pat, norm_text) for pat in route_patterns):
            return result(MedicalIntent.MEDICATION_ROUTE_QUERY, "HISTORY_QUESTION", "medication", "medication_route_form", ui_category="Meds", empathy=empathy_detected)

        # 7G. Medication Duration Query
        duration_patterns = [
            r"\b(how\s+long\s+have\s+you\s+been\s+(taking|on)\s+(this|your|[a-z]+))\b",
            r"\b(when\s+did\s+you\s+start\s+(taking|on)\s+(this|your|[a-z]+))\b",
            r"^how long have you (been taking|taken)\b"
        ]
        if any(re.search(pat, norm_text) for pat in duration_patterns):
            return result(MedicalIntent.MEDICATION_DURATION_QUERY, "HISTORY_QUESTION", "medication", "medication_duration", ui_category="Meds", empathy=empathy_detected)

        # 7H. Medication Name Query
        name_query_patterns = [
            r"^what\s+is\s+this\s+(medicine|medication|drug|pill|tablet)\b",
            r"^which\s+drug\s+is\s+this\b",
            r"^what\s+is\s+(amlodipine|atorvastatin|paracetamol|aspirin|sertraline|escitalopram|albuterol)\b",
            r"^tell\s+me\s+about\s+(amlodipine|atorvastatin|paracetamol|aspirin|sertraline|escitalopram|albuterol)\b"
        ]
        if any(re.search(pat, norm_text) for pat in name_query_patterns):
            return result(MedicalIntent.MEDICATION_NAME_QUERY, "HISTORY_QUESTION", "medication", "medication_name_query", ui_category="Meds", empathy=empathy_detected)

        # 7I. Medication Adherence Query
        adherence_patterns = [
            r"\b(did\s+you\s+take\s+(your\s+)?(morning\s+|this\s+morning\s+)(medicine|medication|meds|tablets?|pills?))\b",
            r"\b(did\s+you\s+take\s+(your\s+)?(medicine|medication|meds|tablets?|pills?)\s+(this\s+morning|today|yesterday|last\s+night))\b",
            r"\b(did\s+you\s+take\s+(your\s+)?morning\s+(medicine|medication|tablets?|pills?|meds))\b",
            r"\b(have\s+you\s+(taken|missed|skipped)\s+(any\s+)?(of\s+your\s+)?(doses?|medications?|medicines?|pills?|tablets?))\b",
            r"\b(miss(ed)?|skip(ped)?|forget|forgotten|forgetting)\s+(any\s+)?(doses?|medications?|medicines?|pills?|tablets?|atorvastatin|amlodipine|[a-z]+)\b",
            r"\b(take|taking)\s+(your\s+)?(medications?|medicines?|pills?|tablets?|drugs?|it|them)\s+regularly\b",
            r"\b(compliant|compliance|adherent|adherence)\s+(with\s+)?(your\s+)?(medications?|medicines?|treatment)\b",
            r"\bhave\s+you\s+been\s+(taking|skipping)\s+your\s+(medications?|tablets?|pills?|medicines?)\b",
            r"\bare\s+you\s+taking\s+your\s+(medications?|medicines?|tablets?|pills?|it|them)\b",
            r"\bdid\s+you\s+miss\s+(your\s+|the\s+)?(atorvastatin|amlodipine|medicine|medication|meds|tablets?|pills?|doses?)\b",
            r"\bhave\s+you\s+been\s+skipping\s+your\s+(tablets?|medications?|pills?|medicines?)\b",
            r"\bdo\s+you\s+forget\s+to\s+take\s+your\s+(medications?|medicines?|tablets?|pills?)\b",
            r"^\s*do\s+you\s+take\s+your\s+(medicine|medication)\s+regularly\b",
            r"^\s*are\s+you\s+taking\s+your\s+(medications?|medicines?)\b",
            r"^\s*did\s+you\s+miss\s+any\s+doses?\b",
            r"^\s*did\s+you\s+miss\s+your\s+(medication|medicine|atorvastatin|amlodipine)\b",
            r"^\s*are\s+you\s+taking\s+it\s+regularly\b",
            r"^\s*did\s+you\s+take\s+your\s+medicine\s+this\s+morning\b",
            r"^\s*did\s+you\s+take\s+your\s+morning\s+medicine\b"
        ]
        if any(re.search(pat, norm_text) for pat in adherence_patterns):
            t_ref = "today" if ("today" in norm_text or "morning" in norm_text) else "current"
            return result(MedicalIntent.MEDICATION_ADHERENCE, "HISTORY_QUESTION", "medication", "medication_adherence", ui_category="Meds", empathy=empathy_detected, temporal=t_ref)

        inhaler_triggers = [
            r"used (your |the )?(blue )?inhaler today",
            r"have you used (your |the )?(blue )?inhaler",
            r"did you use (your |the )?(blue )?inhaler",
            r"how many (times|puffs) did you (use|take) (your |the )?inhaler",
            r"how many times (have you used|did you use) (the |your )?inhaler",
            r"inhaler today",
            r"take your inhaler today",
        ]
        if any(re.search(pat, norm_text) for pat in inhaler_triggers):
            return result(MedicalIntent.MEDICATION_HISTORY, "HISTORY_QUESTION", "medication", "inhaler_use", ui_category="Meds", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 8. GENERAL MEDICATION HISTORY
        # -------------------------------------------------------------
        med_history_patterns = [
            r"\b(what\s+(medications?|medicines?|pills?|drugs?|tablets?|prescriptions?|meds)\s+(do you|are you|did you)\s+(take|use|have|eat|on|prescribed))\b",
            r"\b(which\s+(medications?|medicines?|pills?|drugs?|tablets?)\s+(do you|are you|did you)\s+(take|use|have|eat|on))\b",
            r"\b(are\s+you\s+(taking|on|eating)\s+(any\s+)?(daily\s+|current\s+|regular\s+)?(type\s+of\s+)?(medications?|medicines?|pills?|prescriptions?|tablets?|drugs?))\b",
            r"\b(are\s+you\s+on\s+([a-z\s]+))\b",
            r"\b(did\s+you\s+(take|eat|have)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(have\s+you\s+(taken|eaten|had)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(had\s+you\s+(eat|taken|had)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(eat|ate|taken?)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicine|medication|pill|drug|tablet|meds)\b",
            r"\b(can\s+you\s+tell\s+me\s+what\s+(type\s+of\s+)?(medicine|medication|meds)\s+(do\s+you|did\s+you)\s+(take|eat))\b",
            r"\b(what\s+type\s+of\s+medicine\s+did\s+you\s+(take|eat))\b",
            r"^(did you eat medicine|did you eat medcian|did you take medicine|what medicine do you take|what medications do you take|what meds are you on|what tablets do you take|what pills are you taking|did you take any medicine|did you take your medicine|did you take any meds|can you tell me what medication you take|which medications are you on|what medicine do you take every day)\b"
        ]
        has_med_history_pattern = any(re.search(pat, norm_text) for pat in med_history_patterns)
        has_med_verb_inquiry = (
            len(med_entities) > 0 and any(k in norm_text for k in [
                "eat", "ate", "eating", "take", "taken", "taking", "took", "had", "have", "on", "use", "using",
                "what", "which", "any", "prescript", "daily", "current", "regular", "routine",
                "did you", "have you", "had you", "are you", "do you", "tell me", "normally"
            ])
        )
        if has_med_history_pattern or has_med_verb_inquiry:
            return result(MedicalIntent.MEDICATION_HISTORY, "HISTORY_QUESTION", "medication", "current_medications", ui_category="Meds", empathy=empathy_detected, temporal="current")

        # -------------------------------------------------------------
        # 9. DIET HISTORY
        # -------------------------------------------------------------
        diet_triggers = [
            r"\b(breakfast|lunch|dinner|supper|brunch)\b",
            r"\b(had\s+(you\s+)?breakfast|did you have breakfast|did you eat breakfast|what did you have for (breakfast|lunch|dinner)|what did you eat for (breakfast|lunch|dinner)|what was you eat in your breakfast)\b",
            r"\b(what did you eat (today|yesterday)|what did you have to eat|food intake)\b",
            r"\b(eat|ate|having)\s+(food|meals?|anything to eat|something to eat)\b",
            r"\b(diet|food intake|meals?)\b"
        ]
        has_diet_trigger = any(re.search(pat, norm_text) for pat in diet_triggers) or len(food_entities) > 0
        if has_diet_trigger and len(med_entities) == 0:
            slot = "diet_history"
            if "breakfast" in norm_text:
                slot = "breakfast"
            elif "lunch" in norm_text:
                slot = "lunch"
            elif "dinner" in norm_text or "supper" in norm_text:
                slot = "dinner"
            elif "snack" in norm_text or "snacks" in norm_text:
                slot = "snacks"
            elif any(k in norm_text for k in ["eat", "ate", "food", "meal"]):
                slot = "general_meal"

            time_ref = "unspecified"
            if any(k in norm_text for k in ["yesterday", "last night", "past day", "previous day", "last evening"]):
                time_ref = "previous_day"
            elif any(k in norm_text for k in ["today", "this morning", "this afternoon", "earlier today", "morning"]):
                time_ref = "today"

            return result(MedicalIntent.DIET_HISTORY, "HISTORY_QUESTION", "diet", slot, ui_category="SocialHx", empathy=empathy_detected, temporal=time_ref)

        # -------------------------------------------------------------
        # 10. ALLERGIES & PAST MEDICAL HISTORY & GENDER
        # -------------------------------------------------------------
        if any(k in norm_text for k in ["allerg", "allergic", "drug reaction", "sensitivities", "penicillin"]):
            return result(MedicalIntent.ALLERGIES, "HISTORY_QUESTION", "allergies", "drug_allergies", ui_category="Allergies", empathy=empathy_detected)

        if any(k in norm_text for k in ["medical history", "past medical", "chronic condition", "past illness", "past diseases", "prior medical", "hypertension", "high blood pressure", "cholesterol", "heart problem"]):
            return result(MedicalIntent.PAST_MEDICAL_HISTORY, "HISTORY_QUESTION", "pmh", "past_medical_history", ui_category="PMH", empathy=empathy_detected)

        if any(k in norm_text for k in ["vitiligo", "depigmentation", "white skin patches"]):
            return result(MedicalIntent.PAST_MEDICAL_HISTORY, "HISTORY_QUESTION", "pmh", "vitiligo", ui_category="PMH", empathy=empathy_detected)

        if any(k in norm_text for k in ["cancer", "tumor", "malignancy", "chemotherapy", "radiation therapy"]):
            return result(MedicalIntent.PAST_MEDICAL_HISTORY, "HISTORY_QUESTION", "pmh", "cancer", ui_category="PMH", empathy=empathy_detected)

        if any(k in norm_text for k in ["pcod", "pcos", "polycystic ovary", "polycystic", "menstrual", "period", "periods", "menstruation", "last menstrual period", "lmp", "menses", "pregnant", "pregnancy", "pap smear", "gynecological"]):
            slot = "pcod" if any(p in norm_text for p in ["pcod", "pcos", "polycystic"]) else "menstrual_history"
            return result(MedicalIntent.GENDER_INAPPLICABLE, "HISTORY_QUESTION", "pmh", slot, ui_category="PMH", empathy=empathy_detected)

        if any(k in norm_text for k in ["family history", "father", "mother", "parent", "genetic", "heart disease in your family", "asthma in your family", "runs in your family"]):
            return result(MedicalIntent.FAMILY_HISTORY, "HISTORY_QUESTION", "family_history", "family_cardiac", ui_category="FamilyHx", empathy=empathy_detected)

        if any(k in norm_text for k in ["smoke", "tobacco", "cigarette", "smoking", "vape", "vaping", "alcohol", "drink", "wine", "beer", "work", "job", "occupation", "stress", "drugs", "cocaine", "substance"]):
            return result(MedicalIntent.SOCIAL_HISTORY, "HISTORY_QUESTION", "social_history", "habits_or_occupation", ui_category="SocialHx", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 11. OPQRST DIMENSIONS
        # -------------------------------------------------------------
        # Onset Activity
        onset_act_triggers = [
            r"what were you doing when",
            r"what was you doing when",
            r"what were you doing.*(start|began|happen|occur)",
            r"doing.*when (this|it|the pain|your symptoms) (started|began)",
            r"activity.*(start|began)",
            r"what were you doing at the time",
        ]
        if any(re.search(pat, norm_text) for pat in onset_act_triggers):
            return result(MedicalIntent.ONSET_ACTIVITY, "HISTORY_QUESTION", "symptom", "onset_activity", ui_category="HPI", empathy=empathy_detected)

        # Character
        character_triggers = [
            r"feel like",
            r"what (does|did|do) (the|your|this)? (pain|discomfort|pressure|tightness|chest pain|sensation) feel like",
            r"describe (what |how )?(the |your |what the )?(pain|discomfort|pressure|tightness|sensation|chest pain|breathing)",
            r"how (would you|do you) describe (the |your |this )?(pain|discomfort|pressure|tightness)",
            r"character(ize|istic| of)? (the |your )?(pain|discomfort|chest pain)",
            r"quality of (the |your )?(pain|discomfort|chest pain)",
            r"what (kind|type|nature|sort) of (pain|discomfort|pressure|feeling|sensation)",
            r"(is it|is the pain) (pressure|crushing|squeezing|heavy|sharp|dull|burning|aching|throbbing|tight|stabbing)",
            r"sharp or dull",
            r"crushing or sharp",
            r"elephant",
            r"squeezing or pressure",
            r"tell me about (the |your )?(pain|discomfort)",
            r"describe.*pain",
            r"character.*pain",
            r"what is the (pain|sensation) like",
            r"nature of (the |your )?pain"
        ]
        if any(re.search(pat, norm_text) for pat in character_triggers) or (
            ("describe" in norm_text or "feel like" in norm_text or "what kind" in norm_text or "character" in norm_text or "quality" in norm_text)
            and any(p in norm_text for p in ["pain", "discomfort", "pressure", "tightness", "chest", "sensation", "it"])
        ):
            return result(MedicalIntent.CHARACTER, "HISTORY_QUESTION", "symptom", "pain_character", ui_category="HPI", empathy=empathy_detected)

        # Radiation
        radiation_triggers = [
            r"radiat",
            r"spread",
            r"move anywhere",
            r"move (to|into|down|from)",
            r"moved",
            r"go anywhere",
            r"travel",
            r"shoot (down|up|into|to)",
            r"shooting into",
            r"to your (jaw|arm|arms|shoulder|back|neck|groin)",
            r"down your (arm|arms|leg|legs|back)",
            r"into your (jaw|arm|arms|shoulder|back|neck|groin)",
            r"in your (jaw|arm|arms|shoulder|neck)",
            r"where does (the |your )?pain go",
            r"does (it|the pain) go anywhere"
        ]
        if any(re.search(pat, norm_text) for pat in radiation_triggers) and not any(k in norm_text for k in ["tearing", "ripping"]):
            return result(MedicalIntent.RADIATION, "HISTORY_QUESTION", "symptom", "radiation", ui_category="HPI", empathy=empathy_detected)

        # Severity
        severity_triggers = [
            r"1 to 10",
            r"1-10",
            r"scale of 1",
            r"rate (your |the )?pain",
            r"rate (your |the )?discomfort",
            r"how severe",
            r"severity",
            r"pain score",
            r"out of 10",
            r"out of ten",
            r"how bad (is it|is the pain|is your breathing|is the discomfort)",
            r"scale.*10",
        ]
        if any(re.search(pat, norm_text) for pat in severity_triggers):
            return result(MedicalIntent.SEVERITY, "HISTORY_QUESTION", "symptom", "pain_severity", ui_category="HPI", empathy=empathy_detected)

        # Location
        location_triggers = [
            r"where (is|was|are) (the|your)? (pain|discomfort|pressure|tightness|sensation|stomach pain|stomach|chest pain|trouble)",
            r"where exactly",
            r"point to where",
            r"location of (the)? (pain|discomfort|stomach|tightness)",
            r"where does it hurt",
            r"where.*(hurt|pain|ache|discomfort|tight)",
        ]
        if any(re.search(pat, norm_text) for pat in location_triggers) and not any(k in norm_text for k in ["radiat", "spread", "move", "go"]):
            return result(MedicalIntent.LOCATION, "HISTORY_QUESTION", "symptom", "location", ui_category="HPI", empathy=empathy_detected)

        # Onset Timing
        onset_triggers = [
            r"when did (it|this|the pain|the discomfort|the pressure|your symptoms|it all) (start|begin|happen|occur|first start)",
            r"how long (has|have) (this|it|the pain|your symptoms) (been going on|lasted|been happening|been present)",
            r"how long has it been",
            r"when did you first notice",
            r"what time did (it|the pain) start",
            r"start.*how long",
            r"when.*start",
        ]
        if any(re.search(pat, norm_text) for pat in onset_triggers):
            return result(MedicalIntent.ONSET_TIMING, "HISTORY_QUESTION", "symptom", "onset", ui_category="HPI", empathy=empathy_detected)

        # Timing (Constant / Intermittent)
        if any(k in norm_text for k in ["constant", "come and go", "intermittent", "comes and goes", "continuous", "all the time", "fluctuate", "all day", "timing of the pain"]):
            return result(MedicalIntent.TIMING, "HISTORY_QUESTION", "symptom", "timing", ui_category="HPI", empathy=empathy_detected)

        # Aggravating Factors
        if any(k in norm_text for k in ["make it worse", "makes it worse", "worse with", "aggravat", "triggers the pain", "bring on the pain"]):
            return result(MedicalIntent.AGGRAVATING_FACTORS, "HISTORY_QUESTION", "symptom", "aggravating_factors", ui_category="HPI", empathy=empathy_detected)

        # Relieving Factors
        if any(k in norm_text for k in ["make it better", "makes it better", "reliev", "improve with", "help the pain", "ease the pain", "anything help"]):
            return result(MedicalIntent.RELIEVING_FACTORS, "HISTORY_QUESTION", "symptom", "relieving_factors", ui_category="HPI", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 12. ASSOCIATED SYMPTOMS & ROS
        # -------------------------------------------------------------
        if any(k in norm_text for k in ["short of breath", "shortness of breath", "breathless", "winded", "dyspnea", "catch your breath", "trouble breathing", "hard to breathe", "wheez"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "shortness_of_breath", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["sweat", "sweating", "clammy", "cold sweat", "perspir", "diaphoresis"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "sweating", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["dizzy", "dizziness", "lightheaded", "faint", "pass out", "blackout", "syncope"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "dizziness", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["nausea", "nauseous", "sick to your stomach", "queasy"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "nausea", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["vomit", "vomiting", "throw up", "threw up", "throwing up", "puke", "puking"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "vomiting", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["fever", "feverish", "hot and cold", "chills", "shiver"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "fever", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["palpitation", "heart racing", "fluttering", "fast heart beat", "fast heartbeat"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "palpitations", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["body pain", "body ache", "body aches", "muscle pain", "muscle aches", "myalgia", "joint pain", "hurting all over"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "body_aches", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["cough", "coughing", "phlegm", "sputum"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "cough", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["back pain", "pain in your back", "pain in the back", "back hurt", "back ache", "backache"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "back_pain", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["swelling", "edema", "swollen feet", "swollen ankles", "swollen legs", "fluid in legs", "puffy legs"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "swelling", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["bruis", "bleeding", "easy bruising", "nosebleed", "blood in stool"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "bleeding", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["numbness", "tingling", "pins and needles", "weakness in arm", "weakness in leg"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "neurological", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["diarrhea", "loose stool", "bowel movement"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "diarrhea", ui_category="HPI", empathy=empathy_detected)

        if any(k in norm_text for k in ["headache", "head ache", "head pain", "migraine"]):
            return result(MedicalIntent.ASSOCIATED_SYMPTOM, "HISTORY_QUESTION", "symptom", "headache", ui_category="HPI", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 13. GREETINGS & OPENING CHIEF COMPLAINT
        # -------------------------------------------------------------
        greeting_pattern = r"^(hi+|hello+|hey+|howdy|greetings|good morning|good afternoon|good evening|doctor|dr\b)"
        if re.search(greeting_pattern, norm_text) and len(tokens) <= 6 and not any(k in norm_text for k in ["pain", "start", "what brought", "condition", "inhaler", "breathe", "describe", "feel"]):
            return result(MedicalIntent.GREETING, "GREETING", "general", "greeting", ui_category="General", empathy=empathy_detected)

        chief_complaint_triggers = [
            r"what (brought|brings) you",
            r"how can i help",
            r"tell me (about |what )?(what happened|what is going on|what brings you)",
            r"what is (the matter|going on|wrong|troubling you|the problem)",
            r"what happened",
            r"why (are you|did you come|did you call)",
            r"chief complaint",
        ]
        if any(re.search(pat, norm_text) for pat in chief_complaint_triggers):
            return result(MedicalIntent.OPENING_COMPLAINT, "HISTORY_QUESTION", "symptom", "chief_complaint", ui_category="General", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 14. OFF-TOPIC
        # -------------------------------------------------------------
        unrelated_patterns = [
            r"hairfall", r"hair fall", r"hair loss", r"favorite food",
            r"favorite (movie|color|song|game|sport|actor|dish|hobby|book)",
            r"what is your favorite", r"who (is|was) the president",
            r"tell me a joke", r"weather today", r"stock market", r"cricket", r"football"
        ]
        if any(re.search(pat, norm_text) for pat in unrelated_patterns):
            return result(MedicalIntent.OFF_TOPIC, "OFF_TOPIC", "general", "off_topic", ui_category="General", empathy=empathy_detected)

        # -------------------------------------------------------------
        # 15. FALLBACK / UNKNOWN
        # -------------------------------------------------------------
        return result(MedicalIntent.UNKNOWN, "UNKNOWN", "general", "unknown", ui_category="General", confidence=0.4, empathy=empathy_detected)
