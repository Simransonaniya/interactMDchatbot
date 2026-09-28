"""
InteractMD — Medical Intent Classifier & Slot Extractor.
Combines semantic normalization, lexical analysis, entity recognition,
verb-object dependency logic, and contextual slot extraction.
"""

import re
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from medical_nlu.normalizer import MedicalNormalizer, NormalizedText
from medical_nlu.entity_extractor import MedicalEntityExtractor, MedicalEntity
from medical_nlu.medication_lexicon import MedicationLexicon


class MedicalIntent(str, Enum):
    # Core Medical & Medication Intents
    MEDICATION_HISTORY = "MEDICATION_HISTORY"
    MEDICATION_STATEMENT = "MEDICATION_STATEMENT"
    MEDICATION_NAME_FRAGMENT = "MEDICATION_NAME_FRAGMENT"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
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
class StructuredNLUResult:
    raw_query: str
    normalized_query: str
    intent: MedicalIntent
    primary_type: str
    topic: str
    slot: str
    entities: List[MedicalEntity]
    temporal_reference: str = "unspecified"
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
            "temporal_reference": self.temporal_reference,
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
    Ensures complete semantic interpretation (verb + object + entity composition)
    rather than brittle keyword matching.
    """

    def __init__(self):
        self.normalizer = MedicalNormalizer()
        self.entity_extractor = MedicalEntityExtractor()
        self.lexicon = MedicationLexicon.get_instance()

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

        # Extract previous-turn state if available
        last_topic = None
        last_slot = None
        if session_state:
            last_topic = getattr(session_state, "last_topic", None) or getattr(session_state, "last_question_topic", None)
            last_slot = getattr(session_state, "last_slot", None) or getattr(session_state, "last_disclosed_fact_key", None)

        # -------------------------------------------------------------
        # 0. UNCLEAR / NOISE SHIELD
        # -------------------------------------------------------------
        if re.fullmatch(r"[^a-zA-Z0-9\s]+", raw_low) or raw_low in ["...", ".", "..", "how was it", "what about that", "and then"]:
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.UNCLEAR,
                primary_type="UNKNOWN",
                topic="general",
                slot="unclear",
                entities=entities,
                confidence=0.1,
                ui_category="General"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.PROMPT_INJECTION,
                primary_type="PROMPT_INJECTION",
                topic="security",
                slot="injection_defense",
                entities=entities,
                ui_category="Security"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.DIAGNOSIS_REQUEST,
                primary_type="DIAGNOSIS_REQUEST",
                topic="diagnosis",
                slot="diagnosis_shield",
                entities=entities,
                ui_category="General"
            )

        diag_statement_patterns = [
            r"\b(i think (this is|this may be|this might be|you have|you are having)|this appears to be|this looks like|my impression is|i believe (this is|you have)|could be|may be)\b.*\b(heart attack|cardiac|myocardial|infarction|angina|coronary|anxiety|panic|panic attack|gerd|reflux|acid reflux|costochondritis|muscle strain|pulmonary embolism|pe|asthma|pneumonia)\b",
            r"\b(you (are having|have|might be having)|this is|this may be|it is)\s+(a |an )?(heart attack|cardiac event|cardiac issue|panic attack|angina|stemi|mi)\b",
            r"\b(i suspect|diagnosing you with|working diagnosis is)\b.*\b(cardiac|heart attack|angina|anxiety|gerd)\b",
        ]
        if any(re.search(pat, norm_text) for pat in diag_statement_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.DIAGNOSIS_STATEMENT,
                primary_type="DIAGNOSIS_STATEMENT",
                topic="diagnosis",
                slot="clinician_diagnosis",
                entities=entities,
                ui_category="General"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.EXAM_REQUEST,
                primary_type="EXAM_REQUEST",
                topic="examination",
                slot=target,
                entities=entities,
                action_target=target,
                ui_category="Exam"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.INVESTIGATION_REQUEST,
                primary_type="INVESTIGATION_REQUEST",
                topic="investigations",
                slot=target,
                entities=entities,
                action_target=target,
                ui_category="Diagnostics"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.CHALLENGE,
                primary_type="CHALLENGE",
                topic=last_topic or "general",
                slot=last_slot or "challenge",
                entities=entities,
                ui_category="General",
                is_follow_up=True,
                relationship="CHALLENGE",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

        clarification_patterns = [
            r"^(are you sure|really\??|are you certain|you sure|are you positive|are you absolutely sure|are you really sure)\b",
            r"^(can you explain that again|what do you mean|could you clarify|tell me more about that|could you repeat that|can you explain|can you clarify)\b",
            r"^(what do you mean by that|how is that possible|how can you be sure)\b",
            r"^(are you sure about that|you sure about that)\b",
        ]
        if any(re.search(pat, norm_text) for pat in clarification_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.CLARIFICATION,
                primary_type="CLARIFICATION",
                topic=last_topic or "general",
                slot=last_slot or "clarification",
                entities=entities,
                ui_category="General",
                is_follow_up=True,
                relationship="CLARIFICATION",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

        confirmation_patterns = [
            r"\b(is that correct|is this correct|am i understanding correctly|am i right)\b",
            r"\b(you said.*(correct|right))\b",
            r"^(so (it is|it has been|it is been|you feel|it started|you have|the pain is))\b",
            r"^so\b.*(minutes|hours|days|weeks|\d+|correct|right|going on)\b",
        ]
        if any(re.search(pat, norm_text) for pat in confirmation_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.CONFIRMATION,
                primary_type="CONFIRMATION",
                topic=last_topic or "general",
                slot=last_slot or "confirmation",
                entities=entities,
                ui_category="General",
                is_follow_up=True,
                relationship="CONFIRMATION",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.EMPATHY_REASSURANCE,
                primary_type="EMPATHY_REASSURANCE",
                topic="general",
                slot="empathy",
                entities=entities,
                empathy_detected=True,
                ui_category="General"
            )

        # -------------------------------------------------------------
        # 6. CLINICIAN DIRECTIVES & STATEMENTS (Prescribing / Orders)
        # -------------------------------------------------------------
        med_statement_patterns = [
            r"^(take|try|have)\s+(a\s+|the\s+|this\s+)?(paracetamol|tablet|pill|medicine|medication|aspirin|sertraline|nicip|nicip\s+plus|atorvastatin|amlodipine|inhaler|ibuprofen|tylenol|capsule)\b",
            r"\b(you (should|can|need to|must|have to)|i (will|am going to|can|want to|recommend you)|let us|we (will|should|can))\s+(take|give you|prescribe|administer|try|start you on)\s+(this\s+|some\s+|the\s+|a\s+)?(medicine|medication|pill|drug|tablet|treatment|dose|prescription|paracetamol|aspirin|nicip|nicip\s+plus|atorvastatin|amlodipine)\b",
            r"\b(you should take|take)\s+(tablet|medicine|a tablet|a pill|paracetamol|aspirin|sertraline|nicip\s+plus|atorvastatin|amlodipine)\b",
            r"\b(prescribe|prescribing|order)\s+(medication|medicine|pill|drug|tablet|treatment)\b",
            r"\b(i am giving you|i will give you|let me give you)\s+(some\s+|a\s+|the\s+)?(medicine|medication|pill|tablet|drug|dose)\b",
        ]
        is_interrogative = any(norm_text.startswith(w) for w in ["did you", "do you", "have you", "had you", "are you", "what", "which", "can you tell"])
        
        if not is_interrogative and any(re.search(pat, norm_text) for pat in med_statement_patterns):
            substance = med_entities[0].normalized if med_entities else "medication"
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.MEDICATION_STATEMENT,
                primary_type="MEDICATION_STATEMENT",
                topic="medication",
                slot="treatment_order",
                entities=entities,
                treatment_substance=substance,
                empathy_detected=empathy_detected,
                ui_category="Management"
            )

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
                return StructuredNLUResult(
                    raw_query=raw_query,
                    normalized_query=norm_text,
                    intent=MedicalIntent.MEDICATION_STATEMENT,
                    primary_type="MEDICATION_STATEMENT",
                    topic="medication",
                    slot="treatment_order",
                    entities=entities,
                    treatment_substance=substance,
                    empathy_detected=empathy_detected,
                    ui_category="Management"
                )
            else:
                return StructuredNLUResult(
                    raw_query=raw_query,
                    normalized_query=norm_text,
                    intent=MedicalIntent.MEDICATION_NAME_FRAGMENT,
                    primary_type="MEDICATION_NAME_FRAGMENT",
                    topic="medication",
                    slot="medication_fragment",
                    entities=entities,
                    treatment_substance=substance,
                    empathy_detected=empathy_detected,
                    ui_category="Management"
                )

        # Non-pharmacological Management Statement
        mgmt_patterns = [
            r"\b(you (should|can|need to|must|have to)|let us|try to|i want you to|we will|we shall|let us have you|i would like you to)\s+(take (some |a )?rest|rest|sit down|lie down|relax|stay in bed|stay still|take it easy|stay calm|monitor you|keep you under observation)\b",
            r"\b(sit down and rest|have you sit down|have you lie down|sit down|lie down|take a seat|have a seat|rest for a bit|rest now)\b",
            r"^(you should take (some |a )?rest|take (some |a )?rest|have (some |a )?rest|sit down|lie down|rest now|rest a bit|we will monitor you)\b",
            r"\b(we will monitor you|monitor you)\b",
            r"\b(take (some |a )?rest)\b",
        ]
        if any(re.search(pat, norm_text) for pat in mgmt_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.MANAGEMENT_STATEMENT,
                primary_type="MANAGEMENT_STATEMENT",
                topic="management",
                slot="rest_or_monitoring",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="Management"
            )

        # -------------------------------------------------------------
        # 7. MEDICATION ADHERENCE VS MEDICATION HISTORY
        # -------------------------------------------------------------
        adherence_patterns = [
            r"\b(did you take (your )?(medicine|medication|meds|tablets|pills) today)\b",
            r"\b(have you (taken|missed|skipped) (any )?(of your )?(doses|medications|medicines|pills|tablets))\b",
            r"\b(miss(ed)?|skip(ped)?|forget|forgotten|forgetting)\s+(any\s+)?(doses|medications|medicines|pills|tablets)\b",
            r"\b(taking\s+(your\s+)?(medications|medicines|pills|tablets|drugs)\s+regularly)\b",
            r"\b(compliant|compliance|adherent|adherence)\s+(with\s+)?(your\s+)?(medication|medicine|treatment)\b",
            r"\b(did you take (your )?morning (medicine|medication|tablet|pill))\b",
            r"\b(have you been taking (your )?medications? regularly)\b",
            r"\b(are you taking (your )?medications? regularly)\b",
        ]
        if any(re.search(pat, norm_text) for pat in adherence_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.MEDICATION_ADHERENCE,
                primary_type="HISTORY_QUESTION",
                topic="medication",
                slot="medication_adherence",
                entities=entities,
                temporal_reference="today" if "today" in norm_text or "morning" in norm_text else "current",
                empathy_detected=empathy_detected,
                ui_category="Meds"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.MEDICATION_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="medication",
                slot="inhaler_use",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="Meds"
            )

        med_history_patterns = [
            r"\b(what\s+(medications?|medicines?|pills?|drugs?|tablets?|prescriptions?)\s+(do you|are you|did you)\s+(take|use|have|eat|on))\b",
            r"\b(which\s+(medications?|medicines?|pills?|drugs?|tablets?)\s+(do you|are you|did you)\s+(take|use|have|eat))\b",
            r"\b(are you (taking|on|eating)\s+(any\s+)?(daily\s+|current\s+|regular\s+)?(type of\s+)?(medications?|medicines?|pills?|prescriptions?|tablets?|drugs?))\b",
            r"\b(did you (take|eat|have)\s+(any\s+)?(type of\s+|kind of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(have you (taken|eaten|had)\s+(any\s+)?(type of\s+|kind of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(had you (eat|taken|had)\s+(any\s+)?(type of\s+|kind of\s+)?(medicine|medication|pill|drug|tablet|meds))\b",
            r"\b(eat|ate|taken?)\s+(any\s+)?(type of\s+|kind of\s+)?(medicine|medication|pill|drug|tablet|meds)\b",
            r"\b(can you tell me what (type of )?(medicine|medication|meds) (do you|did you) (take|eat))\b",
            r"\b(what type of medicine did you (take|eat))\b",
            r"^(did you eat medicine|did you eat medcian|did you take medicine|what medicine do you take|what medications do you take|what meds are you on|what tablets do you take|what pills are you taking|did you take any medicine|did you take your medicine|did you take any meds|can you tell me what medication you take)\b"
        ]
        has_med_history_pattern = any(re.search(pat, norm_text) for pat in med_history_patterns)
        has_med_verb_inquiry = (
            len(med_entities) > 0 and any(k in norm_text for k in [
                "eat", "ate", "eating", "take", "taken", "taking", "took", "had", "have", "on", "use", "using",
                "what", "which", "any", "prescript", "daily", "current", "regular", "routine",
                "did you", "have you", "had you", "are you", "do you", "tell me"
            ])
        )
        if has_med_history_pattern or has_med_verb_inquiry:
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.MEDICATION_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="medication",
                slot="current_medications",
                entities=entities,
                temporal_reference="current",
                empathy_detected=empathy_detected,
                ui_category="Meds"
            )

        # -------------------------------------------------------------
        # 8. DIET HISTORY
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

            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.DIET_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="diet",
                slot=slot,
                entities=entities,
                temporal_reference=time_ref,
                empathy_detected=empathy_detected,
                ui_category="SocialHx"
            )

        # -------------------------------------------------------------
        # 9. ALLERGIES & PAST MEDICAL HISTORY & GENDER
        # -------------------------------------------------------------
        if any(k in norm_text for k in ["allerg", "allergic", "drug reaction", "sensitivities", "penicillin"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ALLERGIES,
                primary_type="HISTORY_QUESTION",
                topic="allergies",
                slot="drug_allergies",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="Allergies"
            )

        if any(k in norm_text for k in ["medical history", "past medical", "chronic condition", "past illness", "past diseases", "prior medical", "hypertension", "high blood pressure", "cholesterol", "heart problem"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.PAST_MEDICAL_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="pmh",
                slot="past_medical_history",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="PMH"
            )

        if any(k in norm_text for k in ["vitiligo", "depigmentation", "white skin patches"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.PAST_MEDICAL_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="pmh",
                slot="vitiligo",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="PMH"
            )

        if any(k in norm_text for k in ["cancer", "tumor", "malignancy", "chemotherapy", "radiation therapy"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.PAST_MEDICAL_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="pmh",
                slot="cancer",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="PMH"
            )

        if any(k in norm_text for k in ["pcod", "pcos", "polycystic ovary", "polycystic", "menstrual", "period", "periods", "menstruation", "last menstrual period", "lmp", "menses", "pregnant", "pregnancy", "pap smear", "gynecological"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.GENDER_INAPPLICABLE,
                primary_type="HISTORY_QUESTION",
                topic="pmh",
                slot="pcod" if any(p in norm_text for p in ["pcod", "pcos", "polycystic"]) else "menstrual_history",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="PMH"
            )

        if any(k in norm_text for k in ["family history", "father", "mother", "parent", "genetic", "heart disease in your family", "asthma in your family", "runs in your family"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.FAMILY_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="family_history",
                slot="family_cardiac",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="FamilyHx"
            )

        if any(k in norm_text for k in ["smoke", "tobacco", "cigarette", "smoking", "vape", "vaping", "alcohol", "drink", "wine", "beer", "work", "job", "occupation", "stress", "drugs", "cocaine", "substance"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.SOCIAL_HISTORY,
                primary_type="HISTORY_QUESTION",
                topic="social_history",
                slot="habits_or_occupation",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="SocialHx"
            )

        # -------------------------------------------------------------
        # 10. OPQRST DIMENSIONS
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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ONSET_ACTIVITY,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="onset_activity",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.CHARACTER,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="pain_character",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.RADIATION,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="radiation",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.SEVERITY,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="pain_severity",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.LOCATION,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="location",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ONSET_TIMING,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="onset",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        # Timing (Constant / Intermittent)
        if any(k in norm_text for k in ["constant", "come and go", "intermittent", "comes and goes", "continuous", "all the time", "fluctuate", "all day", "timing of the pain"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.TIMING,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="timing",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        # Aggravating Factors
        if any(k in norm_text for k in ["make it worse", "makes it worse", "worse with", "aggravat", "triggers the pain", "bring on the pain"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.AGGRAVATING_FACTORS,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="aggravating_factors",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        # Relieving Factors
        if any(k in norm_text for k in ["make it better", "makes it better", "reliev", "improve with", "help the pain", "ease the pain", "anything help"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.RELIEVING_FACTORS,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="relieving_factors",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        # -------------------------------------------------------------
        # 11. ASSOCIATED SYMPTOMS & ROS
        # -------------------------------------------------------------
        if any(k in norm_text for k in ["short of breath", "shortness of breath", "breathless", "winded", "dyspnea", "catch your breath", "trouble breathing", "hard to breathe", "wheez"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="shortness_of_breath",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["sweat", "sweating", "clammy", "cold sweat", "perspir", "diaphoresis"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="sweating",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["dizzy", "dizziness", "lightheaded", "faint", "pass out", "blackout", "syncope"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="dizziness",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["nausea", "nauseous", "sick to your stomach", "queasy"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="nausea",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["vomit", "vomiting", "throw up", "threw up", "throwing up", "puke", "puking"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="vomiting",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["fever", "feverish", "hot and cold", "chills", "shiver"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="fever",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["palpitation", "heart racing", "fluttering", "fast heart beat", "fast heartbeat"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="palpitations",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["body pain", "body ache", "body aches", "muscle pain", "muscle aches", "myalgia", "joint pain", "hurting all over"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="body_aches",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["cough", "coughing", "phlegm", "sputum"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="cough",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["back pain", "pain in your back", "pain in the back", "back hurt", "back ache", "backache"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="back_pain",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["swelling", "edema", "swollen feet", "swollen ankles", "swollen legs", "fluid in legs", "puffy legs"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="swelling",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["bruis", "bleeding", "easy bruising", "nosebleed", "blood in stool"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="bleeding",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["numbness", "tingling", "pins and needles", "weakness in arm", "weakness in leg"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="neurological",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["diarrhea", "loose stool", "bowel movement"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="diarrhea",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        if any(k in norm_text for k in ["headache", "head ache", "head pain", "migraine"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.ASSOCIATED_SYMPTOM,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="headache",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="HPI"
            )

        # -------------------------------------------------------------
        # 12. GREETINGS & OPENING CHIEF COMPLAINT
        # -------------------------------------------------------------
        greeting_pattern = r"^(hi+|hello+|hey+|howdy|greetings|good morning|good afternoon|good evening|doctor|dr\b)"
        if re.search(greeting_pattern, norm_text) and len(tokens) <= 6 and not any(k in norm_text for k in ["pain", "start", "what brought", "condition", "inhaler", "breathe", "describe", "feel"]):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.GREETING,
                primary_type="GREETING",
                topic="general",
                slot="greeting",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="General"
            )

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
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.OPENING_COMPLAINT,
                primary_type="HISTORY_QUESTION",
                topic="symptom",
                slot="chief_complaint",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="General"
            )

        # -------------------------------------------------------------
        # 13. OFF-TOPIC
        # -------------------------------------------------------------
        unrelated_patterns = [
            r"hairfall", r"hair fall", r"hair loss", r"favorite food",
            r"favorite (movie|color|song|game|sport|actor|dish|hobby|book)",
            r"what is your favorite", r"who (is|was) the president",
            r"tell me a joke", r"weather today", r"stock market", r"cricket", r"football"
        ]
        if any(re.search(pat, norm_text) for pat in unrelated_patterns):
            return StructuredNLUResult(
                raw_query=raw_query,
                normalized_query=norm_text,
                intent=MedicalIntent.OFF_TOPIC,
                primary_type="OFF_TOPIC",
                topic="general",
                slot="off_topic",
                entities=entities,
                empathy_detected=empathy_detected,
                ui_category="General"
            )

        # -------------------------------------------------------------
        # 14. FALLBACK / UNKNOWN
        # -------------------------------------------------------------
        return StructuredNLUResult(
            raw_query=raw_query,
            normalized_query=norm_text,
            intent=MedicalIntent.UNKNOWN,
            primary_type="UNKNOWN",
            topic="general",
            slot="unknown",
            entities=entities,
            confidence=0.4,
            empathy_detected=empathy_detected,
            ui_category="General"
        )
