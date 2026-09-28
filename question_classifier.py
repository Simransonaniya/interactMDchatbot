"""
InteractMD — Question & Clinical Intent Classifier.
Performs semantic intent classification covering:
- HISTORY_QUESTION (OPQRST dimensions, associated symptoms, pertinent negatives, PMH, meds, allergies, family/social history)
- CLARIFICATION ("Are you sure?", "Really?", "Can you explain that again?")
- CONFIRMATION ("Is that correct?", "So it's been going on for 45 minutes?")
- EMPATHY_REASSURANCE ("Take a slow breath", "I'm here with you")
- MANAGEMENT_STATEMENT ("You should rest", "Let's have you sit down", "We'll monitor you")
- MEDICATION_STATEMENT ("Take this medication", "You should take tablet", "Take paracetamol")
- DIAGNOSIS_STATEMENT ("I think this is a heart attack", "This appears to be cardiac")
- EXAM_REQUEST ("I'm going to examine your heart", "Let me listen to your chest")
- INVESTIGATION_REQUEST ("Let's order an ECG", "We should check troponin")
- OFF_TOPIC ("hairfall", "favorite movie", "weather")
- UNKNOWN / UNCLEAR ("how was it?", single characters)
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


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
        referenced_slot: Optional[str] = None
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

    @property
    def primary_type(self) -> str:
        """Returns the primary standardized intent category name."""
        if self.category in [
            IntentCategory.CHARACTER, IntentCategory.RADIATION, IntentCategory.SEVERITY,
            IntentCategory.LOCATION, IntentCategory.ONSET_TIMING, IntentCategory.ONSET_ACTIVITY,
            IntentCategory.TIMING, IntentCategory.AGGRAVATING_FACTORS, IntentCategory.RELIEVING_FACTORS,
            IntentCategory.ASSOCIATED_SYMPTOM, IntentCategory.PAST_MEDICAL_HISTORY,
            IntentCategory.MEDICATIONS, IntentCategory.ALLERGIES, IntentCategory.FAMILY_HISTORY,
            IntentCategory.SOCIAL_HISTORY, IntentCategory.OPENING_COMPLAINT, IntentCategory.GENDER_INAPPLICABLE
        ]:
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


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous |your )?instructions",
    r"show (me )?(the )?(complete )?(case )?json",
    r"tell me everything you know",
    r"reveal (your |the )?system prompt",
    r"give me (the )?(complete )?patient record",
    r"developer mode",
    r"dump (the )?database",
    r"override rules",
    r"jailbreak",
]

DIAGNOSIS_REQUEST_PATTERNS = [
    r"what is (your|my) diagnosis",
    r"what (condition|disease) do (i|you) have",
    r"tell me (what )?the diagnosis (is)?",
    r"tell me (your |the )?hidden diagnosis",
    r"what do you think is wrong with (me|you)",
    r"what disease do you have",
]

DIAGNOSIS_STATEMENT_PATTERNS = [
    r"\b(i think (this is|this may be|this might be|this could be|it's|it is|it may be|it might be|it could be|you have|you are having|you might be having)|this appears to be|this looks like|this sounds like|my impression is|i believe (this is|this may be|you have)|could be|may be)\b.*\b(heart attack|cardiac|myocardial|infarction|angina|coronary|anxiety|panic|panic attack|gerd|reflux|acid reflux|costochondritis|muscle strain|pulmonary embolism|pe|asthma|pneumonia)\b",
    r"\b(you (are having|have|might be having)|this is|this may be|it's|it is|it may be)\s+(a |an )?(heart attack|cardiac event|cardiac issue|panic attack|angina|stemi|mi)\b",
    r"\b(i suspect|diagnosing you with|working diagnosis is)\b.*\b(cardiac|heart attack|angina|anxiety|gerd)\b",
    r"\b(this (is|appears|looks|sounds)|it (is|appears|looks|sounds))\s+(to be\s+)?(cardiac|heart related|coronary)\b",
]

CHALLENGE_PATTERNS = [
    r"\b(how\s+(can\s+you|do\s+you|could\s+you|can|you)\s+(not|don't|dont|not\s+even)\s+know)\b",
    r"\b(how\s+(can\s+you|do\s+you|could\s+you|can|you)\s+(not|don't|dont)\s+remember)\b",
    r"\b(why\s+(can't\s+you|cant\s+you|can\s+you\s+not|don't\s+you|dont\s+you|you\s+don't|you\s+dont|you\s+not)\s+(know|remember))\b",
    r"\b(why\s+(can't|cant)\s+you\s+remember)\b",
    r"\b(are\s+you\s+(made|mad)\s+(that\s+)?(you\s+)?(don't|dont)\s+know\s+anything)\b",
    r"\b(are\s+you\s+(made|mad)\s+(that\s+)?(you\s+)?(can't|cant)\s+remember\s+anything)\b",
    r"\b((you\s+)?(don't|dont)\s+know\s+anything)\b",
    r"\b((you\s+)?(can't|cant)\s+remember\s+anything)\b",
    r"\b(how\s+is\s+(that|this)\s+possible)\b",
    r"\b(how\s+come\s+you\s+(don't|dont|can't|cant)\s+(know|remember))\b",
    r"\b(you\s+really\s+(don't|dont)\s+know)\b",
    r"\b(how\s+can\s+it\s+be\s+that\s+you\s+(don't|dont)\s+know)\b",
    r"\b(how\s+can\s+you\s+be\s+sure\s+you\s+don't\s+know)\b",
    r"^how can you don't know\b",
    r"^how can you dont know\b",
    r"^how you don't know\b",
    r"^how you dont know\b",
    r"^why you don't know\b",
    r"^why you dont know\b",
    r"^why you don't remember\b",
    r"^why you dont remember\b",
    r"^how you don't remember\b",
    r"^how you dont remember\b",
    r"^are you made you don't know anything\b",
    r"^are you made you dont know anything\b",
    r"^why can't you remember\b",
    r"^why cant you remember\b",
    r"^how can't you remember\b",
    r"^how cant you remember\b",
    r"^you don't know\??$",
    r"^you dont know\??$",
    r"^you don't remember\??$",
    r"^you dont remember\??$",
    r"^how can you not know\b",
    r"^how do you not know\b",
    r"^why don't you know\b",
    r"^why dont you know\b",
]

CLARIFICATION_PATTERNS = [
    r"^(are you sure|really\??|are you certain|you sure|are you positive|are you absolutely sure|are you really sure)\b",
    r"^(can you explain that again|what do you mean|could you clarify|tell me more about that|could you repeat that|can you explain|can you clarify)\b",
    r"^(what do you mean by that|how is that possible|how can you be sure)\b",
    r"^(are you sure about that|you sure about that)\b",
    r"^(why\??|what do you mean\??)$",
]

CONFIRMATION_PATTERNS = [
    r"\b(is that correct|is this correct|am i understanding correctly|am i right)\b",
    r"\b(you said.*(correct|right))\b",
    r"^(so it's \d+|so it has been \d+|so you feel \d+|so it started \d+)\b",
    r"^(so (it's|it has been|you have|the pain is))\b",
]

INVESTIGATION_RESULT_SHIELD_PATTERNS = [
    r"what did (my |the )?(stat )?(12-lead )?ecg (show|say)",
    r"what does the ecg show",
    r"what did (my |the )?blood(work| test|s)? (show|say)",
    r"what (is|are) my (troponin|labs|lab results|enzymes)",
    r"what did (my |the )?(chest )?(x-ray|cxr|ct scan|ultrasound) (show|say)",
    r"result of (the )?(ecg|labs|troponin|imaging)",
]

EXAM_ACTION_PATTERNS = [
    r"(i'd like to|i want to|let me|can i|i will) (check|take|examine|listen to|perform|do) (your )?(vital signs|vitals|blood pressure|heart rate|pulse|temp|temperature)",
    r"(i'd like to|i want to|let me|can i|i will|let's) (perform|do|examine) (a |an )?(cardiovascular|respiratory|chest|abdominal|physical|neuro) (exam|examination)",
    r"(i'd like to|i want to|let me|can i|i will|let's) (listen to|auscultate|examine) (your )?(heart|lungs|chest|breathing|abdomen)",
    r"^let's examine (your )?(heart|lungs|chest|breathing|abdomen)",
]

INVESTIGATION_ORDER_PATTERNS = [
    r"(i'd like to|i want to|let's|can we|i will|order|we should) (order|run|get|perform|obtain|check) (a |an )?(stat )?(12-lead )?(ecg|ekg|chest x-ray|cxr|blood test|labs|troponin|ct scan|ultrasound)",
]

# Known Medication Keywords & Entities
KNOWN_MEDICATION_ENTITIES = [
    "paracetamol", "paracetomol", "aspirin", "sertraline", "sertrakine", "escitalopram",
    "escita;pram", "nitro", "nitroglycerin", "heparin", "morphine", "metoprolol",
    "statin", "atorvastatin", "amlodipine", "lisinopril", "inhaler", "tylenol", "ibuprofen",
    "advil", "crocin", "dolo", "combiflam", "nicip", "nicip plus", "niciplus", "nishchit",
    "nishchit plus", "pantocid", "pan 40", "omeprazole", "azithromycin", "antibiotic",
    "antibiotics", "tablet", "tablets", "pill", "pills", "capsule", "capsules", "medicine", "medication"
]

MEDICATION_FRAGMENT_PATTERNS = [
    r"^(paracetamol|paracetomol|aspirin|sertraline|niciplus|nicip|nicip plus|nishchit plus|crocin|dolo|combiflam|atorvastatin|amlodipine|lisinopril|metoprolol|inhaler|tylenol|ibuprofen)(\s+(tablet|tablets|pill|pills|capsule|mg|\d+mg|dose))?$",
    r"^(tablet|tablets|pill|pills|capsule|medicine|medication)$",
    r"^(paracetamol|paracetomol|aspirin|niciplus|nicip plus|crocin|dolo)\s+tablet$",
]

# Medication Statement Triggers (Clinician prescribing / giving medication to patient)
MEDICATION_STATEMENT_PATTERNS = [
    r"\b(take|have|give|giving you|prescribe|prescribing|start you on|start on|administer|try)\b.*\b(paracetamol|paracetomol|tablet|pill|medicine|medication|capsule|aspirin|nitro|nitroglycerin|sertraline|sertrakine|escitalopram|escita;pram|statin|atorvastatin|metoprolol|beta blocker|morphine|heparin|clopidogrel|plavix|inhaler|antibiotic|antibiotics|tylenol|ibuprofen|advil|niciplus|nicip|nicip plus|nishchit plus)\b",
    r"\b(you (can|cab|may|should|need to|must|have to)|i (will|am going to|can|want to|recommend you)|let's|let us|we (will|should|can|are going to))\s+(take|give you|prescribe|administer|try|start you on)\s+(this |some |the |a )?(medicin[a-z]*|pill|drug|tablet|treatment|dose|prescription|remedy|paracetamol|paracetomol|aspirin|niciplus|nicip|nishchit)\b",
    r"\b(you (can|cab)|take this)\b.*\b(medicin|medication|tablet|pill|capsule|drug|sertrakine|sertraline|escitalopram|escita;pram|inhaler|paracetamol|paracetomol|niciplus|nicip plus|nishchit plus)\b",
    r"\b(prescribe|prescribing|order)\s+(medication|medicine|pill|drug|tablet|treatment)\b",
    r"\b(i am giving you|i will give you|i'm giving you|let me give you)\s+(some |a |the )?(medicin|medication|pill|tablet|drug|shot|injection|dose)\b",
    r"^(take|try|have)\s+(a |the |this )?(paracetamol|paracetomol|aspirin|tablet|pill|medicine|medication|tylenol|ibuprofen|niciplus|nicip plus|nishchit plus)\b",
    r"^you should take (tablet|medicine|a tablet|a pill|paracetamol|paracetomol|aspirin|sertraline|niciplus|nicip plus)",
]

# Non-pharmacological Management Instructions (Rest, position, monitoring)
MANAGEMENT_STATEMENT_PATTERNS = [
    r"\b(you (should|can|need to|must|have to)|let's|let us|try to|i want you to|we will|we'll|let's have you|i'd like you to)\s+(take (some |a )?rest|rest|sit down|lie down|relax|stay in bed|stay still|take it easy|stay calm|monitor you|keep you under observation)\b",
    r"\b(sit down and rest|have you sit down|have you lie down|sit down|lie down|take a seat|have a seat|rest for a bit|rest now)\b",
    r"^(you should take (some |a )?rest|take (some |a )?rest|have (some |a )?rest|sit down|lie down|rest now|rest a bit|we will monitor you|we'll monitor you|let's monitor you)\b",
    r"\b(we('ll| will| are going to) monitor you)\b",
    r"\b(take (some |a )?rest)\b",
]

EMPATHY_KEYWORDS = [
    "sorry", "concern", "take care", "take good care", "help you", "comfortable",
    "breathe", "stay calm", "don't worry", "take your time", "here for you",
    "make you comfortable", "must be frightening", "understand", "we are going to take care",
    "i hear you", "you are safe", "we'll figure this out", "in good hands",
    "take a breath", "take a deep breath", "take a slow breath", "i'm here with you",
    "i am here with you", "you're going to be okay", "you will be okay"
]

GENUINELY_UNCLEAR_PATTERNS = [
    r"^how was it\??$",
    r"^what about that\??$",
    r"^and then\??$",
    r"^why\??$",
    r"^what\??$",
    r"^[a-z]{1,3}$",
]

def normalize_message(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^\w\s\?\-\/]", " ", q)
    typo_map = {
        r"\bparacetomol\b": "paracetamol",
        r"\bniciplus\b": "nicip plus",
        r"\bnishchit\s+plus\b": "nicip plus",
        r"\bnishchit\b": "nicip plus",
        r"\bsertrakine\b": "sertraline",
        r"\bescita;pram\b": "escitalopram",
        r"\btabs?\b": "tablet",
        r"\bpills?\b": "pill",
        r"\bmeds?\b": "medicines",
        r"\bmedicians?\b": "medicine",
        r"\bmedecines?\b": "medicine",
        r"\bmedicnes?\b": "medicine",
        r"\bmedicens?\b": "medicine",
        r"\bmedecins?\b": "medicine",
        r"\bmedications?\b": "medicine",
    }
    for pat, repl in typo_map.items():
        q = re.sub(pat, repl, q)
    return re.sub(r"\s+", " ", q).strip()

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


class QuestionClassifier:

    @staticmethod
    def classify(query_text: str, session_state: Optional[Any] = None) -> ClassifiedIntent:
        raw_norm = normalize_message(query_text)
        query = query_text.lower().strip()
        tokens = query.split()

        # Extract previous-turn state if available
        last_topic = None
        last_slot = None
        if session_state:
            last_topic = getattr(session_state, "last_topic", None) or getattr(session_state, "last_question_topic", None)
            last_slot = getattr(session_state, "last_slot", None) or getattr(session_state, "last_disclosed_fact_key", None)

        # 1. Prompt Injection Shield
        if any(re.search(pat, query) for pat in PROMPT_INJECTION_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PROMPT_INJECTION,
                ui_category="Security",
                normalized_query=raw_norm
            )

        # 2. Hidden Diagnosis Request (Doctor asking patient what condition patient has)
        if any(re.search(pat, query) for pat in DIAGNOSIS_REQUEST_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.DIAGNOSIS_REQUEST,
                subconcept="diagnosis_shield",
                ui_category="General",
                normalized_query=raw_norm
            )

        # 3. Diagnosis Statement (Doctor explaining/giving provisional diagnosis to patient)
        if any(re.search(pat, query) for pat in DIAGNOSIS_STATEMENT_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.DIAGNOSIS_STATEMENT,
                subconcept="clinician_diagnosis_statement",
                ui_category="General",
                normalized_query=raw_norm
            )

        # 4. Hidden Investigation Result Shield
        if any(re.search(pat, query) for pat in INVESTIGATION_RESULT_SHIELD_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.INVESTIGATION_REQUEST,
                subconcept="investigation_result_shield",
                ui_category="Diagnostics",
                normalized_query=raw_norm
            )

        # 5. Examination Request
        if any(re.search(pat, query) for pat in EXAM_ACTION_PATTERNS):
            target = "vitals" if any(v in query for v in ["vital", "blood pressure", "pulse", "temp"]) else "physical_exam"
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.EXAMINATION_REQUEST,
                action_target=target,
                ui_category="Exam",
                normalized_query=raw_norm
            )

        # 6. Investigation Order Request
        if any(re.search(pat, query) for pat in INVESTIGATION_ORDER_PATTERNS):
            target = "ecg" if "ecg" in query or "ekg" in query else "labs"
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.INVESTIGATION_REQUEST,
                action_target=target,
                ui_category="Diagnostics",
                normalized_query=raw_norm
            )

        # 7. Challenge Request ("How can you not know?", "How can you don't know", "Why can't you remember?")
        if any(re.search(pat, query) for pat in CHALLENGE_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.CHALLENGE,
                subconcept="challenge",
                ui_category="General",
                normalized_query=raw_norm,
                is_follow_up_to_previous_turn=True,
                relationship="CHALLENGE",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

        # 8. Clarification Request ("Are you sure?", "Really?", "Can you explain that again?")
        if any(re.search(pat, query) for pat in CLARIFICATION_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.CLARIFICATION,
                subconcept="clarification",
                ui_category="General",
                normalized_query=raw_norm,
                is_follow_up_to_previous_turn=True,
                relationship="CLARIFICATION",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

        # 9. Confirmation Request ("Is that correct?", "So it's been 45 minutes?")
        if any(re.search(pat, query) for pat in CONFIRMATION_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.CONFIRMATION,
                subconcept="confirmation",
                ui_category="General",
                normalized_query=raw_norm,
                is_follow_up_to_previous_turn=True,
                relationship="CONFIRMATION",
                referenced_topic=last_topic,
                referenced_slot=last_slot
            )

        # Check if this is a clinician instruction/management statement (prescribing, telling patient to take/rest)
        is_clinician_instruction = (
            any(re.search(pat, query) for pat in MEDICATION_STATEMENT_PATTERNS)
            or any(re.search(pat, raw_norm) for pat in MEDICATION_STATEMENT_PATTERNS)
            or any(re.search(pat, query) for pat in MANAGEMENT_STATEMENT_PATTERNS)
            or any(re.search(pat, raw_norm) for pat in MANAGEMENT_STATEMENT_PATTERNS)
        )

        # 8b. History Question about Medications (Asking what patient takes / ate medicine)
        med_history_patterns = [
            r"\b(had\s+(you\s+)?(eat|taken?|had)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicin[a-z]*|pill[a-z]*|drug[a-z]*|tablet[a-z]*|treatment))\b",
            r"\b(did\s+you\s+(take|eat|have)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicin[a-z]*|pill[a-z]*|drug[a-z]*|tablet[a-z]*))\b",
            r"\b(have\s+you\s+(taken?|eaten|had)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicin[a-z]*|pill[a-z]*|drug[a-z]*|tablet[a-z]*))\b",
            r"\b(what\s+(medications|medicines|pills|drugs|tablets)\s+(do\s+you|are\s+you|did\s+you)\s+(take|use|have|eat))\b",
            r"\b(are\s+you\s+(taking|on|eating)\s+(any\s+)?(daily\s+|current\s+)?(type\s+of\s+)?(medications|medicines|pills|prescriptions|tablets|drugs))\b",
            r"\b(any\s+(current\s+|daily\s+)?(type\s+of\s+)?(medications|medicines|pills|prescriptions|tablets|drugs))\b",
            r"\b(eat|ate|taken?)\s+(any\s+)?(type\s+of\s+|kind\s+of\s+)?(medicin[a-z]*|pill[a-z]*|drug[a-z]*|tablet[a-z]*)\b",
            r"^(had you eat any medicine|had you eat any type of medician|did you take any medicine|what medicines do you take|what medicines you take)\b"
        ]
        if not is_clinician_instruction and (
            any(re.search(pat, query) for pat in med_history_patterns) or any(re.search(pat, raw_norm) for pat in med_history_patterns) or (
                any(k in query or k in raw_norm for k in [
                    "had you eat any medicine", "had you eat any type of medician", "eat any type of medician", "eat any type of medicine",
                    "what medicines do you take", "what medications do you take", "current medications", "any type of medicine", "any type of medician"
                ])
            )
        ):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATIONS,
                subconcept="medications",
                slots=["medications"],
                ui_category="Meds",
                normalized_query=raw_norm
            )

        # 9. Medication Statement from Doctor (Checked before symptom inquiries)
        is_empathy_take = any(k in query for k in ["take care", "take good care", "take your time", "take a deep breath", "take a breath", "take a slow breath", "take a seat"])
        if not is_empathy_take and any(re.search(pat, query) for pat in MEDICATION_STATEMENT_PATTERNS):
            substance = "medication"
            for drug in KNOWN_MEDICATION_ENTITIES:
                if drug in query or drug in raw_norm:
                    substance = drug
                    break
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATION_STATEMENT,
                subconcept="medication_instruction",
                ui_category="Management",
                treatment_substance=substance,
                normalized_query=raw_norm
            )

        # 9b. Medication Name Fragment / Standalone Medication Input
        clean_clean = re.sub(r"[^\w\s]", "", query).strip()
        is_med_fragment = any(re.search(pat, clean_clean) for pat in MEDICATION_FRAGMENT_PATTERNS) or (
            len(tokens) <= 3 and any(drug == clean_clean or f"{drug} tablet" == clean_clean or f"take {drug}" == clean_clean for drug in KNOWN_MEDICATION_ENTITIES)
        )
        if is_med_fragment:
            substance = "medication"
            for drug in KNOWN_MEDICATION_ENTITIES:
                if drug in query or drug in raw_norm:
                    substance = drug
                    break
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATION_NAME_FRAGMENT,
                subconcept="medication_fragment",
                ui_category="Management",
                treatment_substance=substance,
                normalized_query=raw_norm
            )

        # 10. Non-pharmacological Management Statement (Rest, position, monitoring)
        if any(re.search(pat, query) for pat in MANAGEMENT_STATEMENT_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MANAGEMENT_STATEMENT,
                subconcept="rest_or_monitoring",
                ui_category="Management",
                normalized_query=raw_norm
            )

        # 11. Empathy Detection & Bedside Reassurance Statement
        empathy_detected = any(phrase in query for phrase in EMPATHY_KEYWORDS)
        clinical_keywords = [
            "pain", "when did", "where is", "rate", "scale of 1", "sweat", "short of breath", "nausea", "fever", "history",
            "medicine", "allerg", "vomit", "body ache", "cough", "diarrhea", "inhaler", "start", "condition", "describe", "feel"
        ]
        if empathy_detected and len(tokens) <= 25 and not any(k in query for k in clinical_keywords):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.EMPATHY_REASSURANCE,
                ui_category="General",
                empathy_detected=True
            )

        # 12. Greetings
        greeting_pattern = r"^(hi+|hello+|hey+|howdy|greetings|good morning|good afternoon|good evening|doctor|dr\b)"
        is_greeting = bool(re.search(greeting_pattern, query)) or query.strip() in ["hi", "hii", "hiii", "hello", "helloo", "hey", "heyy"]
        if is_greeting and len(tokens) <= 6 and not any(k in query for k in ["pain", "start", "what brought", "condition", "inhaler", "breathe", "describe", "feel"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.GREETING,
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 13. "How are you feeling?" / Small Talk
        if re.search(r"how are you (feeling|doing)|how do you feel today|how are you\b", query) and len(tokens) <= 6:
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SMALL_TALK,
                subconcept="feeling_state",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 14. Open-Ended Chief Complaint ("What brought you in today?", "Why did you come?", etc.)
        chief_complaint_triggers = [
            r"what (brought|brings) you",
            r"how can i help",
            r"tell me (about |what )?(what happened|what's going on|what brings you)",
            r"what('s| is) (the matter|going on|wrong|troubling you|the problem)",
            r"what happened",
            r"why (are you|did you come|did you call)",
            r"chief complaint",
        ]
        if any(re.search(pat, query) for pat in chief_complaint_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.OPENING_COMPLAINT,
                subconcept="chief_complaint",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 15. Out-of-Scope / Off-Topic Statements
        unrelated_patterns = [
            r"hairfall",
            r"hair fall",
            r"hair loss",
            r"hair falling",
            r"favorite food",
            r"favorite (movie|color|song|game|sport|actor|dish|hobby|book)",
            r"what is your favorite",
            r"who (is|was) the president",
            r"tell me a joke",
            r"weather today",
            r"stock market",
            r"cricket",
            r"football",
            r"play a song",
        ]
        if any(re.search(pat, query) for pat in unrelated_patterns):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.OFF_TOPIC,
                subconcept="unrelated_statement",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 16. Gender-Inapplicable Questions (PCOD, menstrual history, pregnancy for male patients)
        if any(k in query for k in ["pcod", "pcos", "polycystic ovary", "polycystic ovarian", "polycystic", "menstrual", "period", "periods", "menstruation", "last menstrual period", "lmp", "menses", "pregnant", "pregnancy", "pap smear", "gynecological"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.GENDER_INAPPLICABLE,
                subconcept="pcod" if any(p in query for p in ["pcod", "pcos", "polycystic"]) else "menstrual_history",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        # 17. OPQRST: Character & Pain Quality
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
        if any(re.search(pat, query) for pat in character_triggers) or (
            ("describe" in query or "feel like" in query or "what kind" in query or "character" in query or "quality" in query)
            and any(p in query for p in ["pain", "discomfort", "pressure", "tightness", "chest", "sensation", "it"])
        ):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.CHARACTER,
                subconcept="character",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 18. OPQRST: Radiation & Spread
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
        if any(re.search(pat, query) for pat in radiation_triggers) and not any(k in query for k in ["tearing", "ripping"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.RADIATION,
                subconcept="radiation",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 19. OPQRST: Severity & Pain Scale
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
        if any(re.search(pat, query) for pat in severity_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SEVERITY,
                subconcept="severity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 20. OPQRST: Location
        location_triggers = [
            r"where (is|was|are) (the|your)? (pain|discomfort|pressure|tightness|sensation|stomach pain|stomach|chest pain|trouble)",
            r"where exactly",
            r"point to where",
            r"location of (the)? (pain|discomfort|stomach|tightness)",
            r"where does it hurt",
            r"where.*(hurt|pain|ache|discomfort|tight)",
        ]
        if any(re.search(pat, query) for pat in location_triggers) and not any(k in query for k in ["radiat", "spread", "move", "go"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.LOCATION,
                subconcept="location",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 21. OPQRST: Onset Timing & Progression
        onset_triggers = [
            r"when did (this|the|it|your|this problem|the pain|this pain|the breathing|this breathing|the shortness of breath|breathing difficulty|your breathing difficulty) (start|begin)",
            r"when did (it|this) (start|begin)",
            r"how long ago did (this|it|the pain|the breathing difficulty) (start|begin)",
            r"when did your breathing (difficulty|trouble|problem) start",
            r"when did (the |your )?(symptoms?|shortness of breath|tightness|pain) start",
            r"how long ago",
            r"how long has (this|it|the pain|this problem|your breathing difficulty|the shortness of breath) (been going on|lasted)",
            r"how many (minutes|hours|days) has this",
            r"time of onset",
        ]
        if any(re.search(pat, query) for pat in onset_triggers) or (
            any(k in query for k in ["when did", "how long ago", "time of onset"]) and any(s in query for s in ["start", "begin", "started", "began", "going on", "lasted"])
        ):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ONSET_TIMING,
                subconcept="onset_timing",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        onset_progression_triggers = [
            r"sudden(ly)? or gradual(ly)?",
            r"gradual(ly)? or sudden(ly)?",
            r"(did it|did this) start (suddenly|gradually)",
            r"come on (suddenly|gradually|all of a sudden)",
            r"was it (sudden|gradual|abrupt)",
            r"(did the |did your )?(symptoms?|breathing|pain) come on (suddenly|gradually)",
            r"onset (sudden|gradual)",
        ]
        if any(re.search(pat, query) for pat in onset_progression_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ONSET_TIMING,
                subconcept="onset_progression",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Activity at onset
        activity_triggers = [
            r"what were you doing when (it|this|the pain|the breathing) (started|began)",
            r"what was happening when (it|this) (started|began)",
            r"where were you when (it|this) (started|began)",
            r"what brought (this|it) on",
            r"activity (at|when) onset",
            r"when it first came on, what were you doing",
        ]
        if any(re.search(pat, query) for pat in activity_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ONSET_ACTIVITY,
                subconcept="onset_activity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Timing / Continuity
        continuity_triggers = [
            r"continuous",
            r"constant",
            r"come and go",
            r"in waves",
            r"steady",
            r"has it been continuous",
            r"has the pain been continuous",
            r"has it gone away",
            r"gone away at all",
            r"intermittent",
        ]
        if any(re.search(pat, query) for pat in continuity_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.TIMING,
                subconcept="timing",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 22. Aggravating / Relieving Factors
        if any(k in query for k in ["better", "reliev", "resting help", "takes the pain away", "what helps"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.RELIEVING_FACTORS,
                subconcept="relieving_factors",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )
        if any(k in query for k in ["worse", "aggravat", "exertion", "moving around", "trigger", "makes it worse", "what makes it"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.AGGRAVATING_FACTORS,
                subconcept="aggravating_factors",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 23. Past Medical History
        pmh_triggers = [
            r"medical condition",
            r"health condition",
            r"medical history",
            r"past medical",
            r"underlying condition",
            r"chronic condition",
            r"chronic illness",
            r"past illness",
            r"diagnosed with any(thing)?",
            r"diagnosed before",
            r"prior medical",
            r"past health",
            r"hospital before",
            r"prior hospital",
            r"surgeries",
            r"past surgeries",
            r"surgery in the past",
            r"history of any disease",
            r"any other (medical |health )?problems",
        ]
        if any(re.search(pat, query) for pat in pmh_triggers) or (
            any(k in query for k in ["medical condition", "medical conditions", "health condition", "health conditions", "medical history", "past medical", "underlying condition", "underlying conditions"])
        ):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PAST_MEDICAL_HISTORY,
                subconcept="past_medical_history",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["vitiligo", "depigmentation", "white skin patches", "skin pigment"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PAST_MEDICAL_HISTORY,
                subconcept="vitiligo",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["cancer", "tumor", "tumour", "malignancy", "malignant", "chemotherapy", "radiation therapy", "carcinoma", "leukemia", "lymphoma", "oncology"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PAST_MEDICAL_HISTORY,
                subconcept="cancer",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        # 24. Allergies & Medications
        if any(k in query for k in ["allerg", "allergic", "drug reaction", "sensitivities", "penicillin"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ALLERGIES,
                subconcept="allergies",
                ui_category="Allergies",
                empathy_detected=empathy_detected
            )

        # Inhaler Specific Inquiry
        inhaler_triggers = [
            r"used (your |the )?(blue )?inhaler today",
            r"have you used (your |the )?(blue )?inhaler",
            r"did you use (your |the )?(blue )?inhaler",
            r"how many (times|puffs) did you (use|take) (your |the )?inhaler",
            r"how many times (have you used|did you use) (the |your )?inhaler",
            r"inhaler today",
            r"take your inhaler today",
        ]
        if any(re.search(pat, query) for pat in inhaler_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATIONS,
                subconcept="inhaler_use",
                ui_category="Meds",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["medicat", "medicine", "pill", "prescription", "inhaler", "taking daily", "what medications", "what pills", "take any medicine", "take any meds", "what drugs"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATIONS,
                subconcept="medications",
                ui_category="Meds",
                empathy_detected=empathy_detected
            )

        # 25. Family History
        if any(k in query for k in ["family history", "father", "mother", "parent", "genetic", "heart disease in your family", "asthma in your family", "runs in your family"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.FAMILY_HISTORY,
                subconcept="family_history",
                ui_category="FamilyHx",
                empathy_detected=empathy_detected
            )

        # 26. Social History
        if any(k in query for k in ["what did you do last weekend", "what did you do on the weekend", "last weekend", "what did you do yesterday", "did you do anything last weekend", "activities last weekend"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SOCIAL_HISTORY,
                subconcept="past_activities",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )

        # 26. Social History - Diet / Meals / Breakfast / Lunch / Dinner
        has_med_terms = any(w in query or w in raw_norm for w in [
            "medicin", "medication", "pill", "tablet", "capsule", "inhaler", "prescript",
            "amlodipine", "atorvastatin", "aspirin", "paracetamol", "niciplus", "nicip",
            "crocin", "dolo", "combiflam", "sertraline", "tylenol", "ibuprofen"
        ])
        diet_triggers = [
            r"\b(breakfast|lunch|dinner|supper|meal|meals|food|eat|ate|eating|eaten|diet|snack|snacks|brunch)\b",
            r"\b(had\s+(you\s+)?breakfast|did you have breakfast|what did you eat|what was you eat|what have you eaten)\b",
            r"\b(what did you have for|what was you eat in your breakfast|have you eaten|had you breakfast)\b",
            r"^(had you breakfast|did you eat breakfast|what was you eat|what did you eat)\b"
        ]
        if not has_med_terms and (any(re.search(pat, query) for pat in diet_triggers) or any(k in query for k in [
            "food eaten", "what did you eat", "eat yesterday", "food yesterday",
            "dinner yesterday", "meal yesterday", "diet history", "last meal",
            "what did you have for dinner", "what did you have for lunch", "what did you have for breakfast",
            "food intake yesterday", "diet yesterday", "what did you eat today", "food intake", "breakfast", "had you breakfast"
        ])):
            diet_slots, time_ref = extract_diet_slots_and_time(query)
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SOCIAL_HISTORY,
                subconcept="diet_history",
                slots=diet_slots,
                time_reference=time_ref,
                ui_category="SocialHx",
                empathy_detected=empathy_detected,
                normalized_query=raw_norm
            )

        if any(k in query for k in ["smoke", "tobacco", "cigarette", "smoking", "vape", "vaping", "alcohol", "drink", "wine", "beer", "work", "job", "occupation", "stress", "drugs", "cocaine", "substance"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SOCIAL_HISTORY,
                subconcept="social_history",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )

        # 27. Associated Symptoms
        if any(k in query for k in ["short of breath", "shortness of breath", "breathless", "winded", "dyspnea", "catch your breath", "trouble breathing", "hard to breathe"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="shortness_of_breath",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["sweat", "sweating", "clammy", "cold sweat", "perspir", "diaphoresis"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="sweating",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["dizzy", "dizziness", "lightheaded", "faint", "pass out", "blackout", "syncope"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="dizziness",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["nausea", "nauseous", "nausious", "sick to your stomach", "queasy"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="nausea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["vomit", "vomiting", "throw up", "threw up", "throwing up", "puke", "puking"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="vomiting",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["fever", "feverish", "hot and cold", "chills", "shiver"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="fever",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["palpitation", "heart racing", "fluttering", "fast heart beat", "fast heartbeat"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="palpitations",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["body pain", "body ache", "body aches", "muscle pain", "muscle aches", "myalgia", "joint pain", "hurting all over"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="body_aches",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["cough", "coughing", "phlegm", "sputum"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="cough",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["wheez", "whistling sound"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="wheezing",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["back pain", "pain in your back", "pain in the back", "back hurt", "back ache", "backache"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="back_pain",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["swelling", "edema", "swollen feet", "swollen ankles", "swollen legs", "fluid in legs", "puffy legs"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="swelling",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["bruis", "bleeding", "easy bruising", "nosebleed", "blood in stool"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="bleeding",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["numbness", "tingling", "pins and needles", "weakness in arm", "weakness in leg"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="neurological",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["diarrhea", "loose stool", "bowel movement"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="diarrhea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        if any(k in query for k in ["headache", "head ache", "head pain", "migraine"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="headache",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 28. Genuinely Unclear Check
        if any(re.fullmatch(pat, query) for pat in GENUINELY_UNCLEAR_PATTERNS) or re.fullmatch(r"[^a-zA-Z0-9\s]+", query):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.UNCLEAR,
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 29. Fallback: Unclassified clinical inquiry
        return ClassifiedIntent(
            raw_query=query_text,
            category=IntentCategory.ASSOCIATED_SYMPTOM,
            subconcept="general_inquiry",
            ui_category="General",
            empathy_detected=empathy_detected
        )
