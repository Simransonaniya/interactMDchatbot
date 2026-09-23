"""
InteractMD — Question & Clinical Intent Classifier.
Performs semantic intent classification covering OPQRST dimensions, associated symptoms,
pertinent negatives, PMH, meds, allergies, family/social history, greetings, empathy,
examinations, investigations, diagnosis inquiries, and prompt injection attempts.
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List


class IntentCategory(str, Enum):
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
    EMPATHY = "EMPATHY"
    EXAMINATION_REQUEST = "EXAMINATION_REQUEST"
    INVESTIGATION_REQUEST = "INVESTIGATION_REQUEST"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
    MANAGEMENT_REQUEST = "MANAGEMENT_REQUEST"
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
        ui_category: str = "General",
        empathy_detected: bool = False,
        jargon_detected: Optional[str] = None,
        action_target: Optional[str] = None
    ):
        self.raw_query = raw_query
        self.category = category
        self.subconcept = subconcept
        self.ui_category = ui_category
        self.empathy_detected = empathy_detected
        self.jargon_detected = jargon_detected
        self.action_target = action_target

    def __repr__(self):
        return f"<ClassifiedIntent category={self.category.value} subconcept={self.subconcept} empathy={self.empathy_detected}>"


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

DIAGNOSIS_PATTERNS = [
    r"what is (your|my) diagnosis",
    r"what (condition|disease) do (i|you) have",
    r"tell me (what )?the diagnosis (is)?",
    r"tell me (your |the )?hidden diagnosis",
    r"what do you think is wrong with (me|you)",
    r"what disease do you have",
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
    r"(i'd like to|i want to|let me|can i|i will) (perform|do) (a |an )?(cardiovascular|respiratory|chest|abdominal|physical|neuro) (exam|examination)",
    r"(i'd like to|i want to|let me|can i|i will) (listen to|auscultate) (your )?(heart|lungs|chest|breathing|abdomen)",
]

INVESTIGATION_ORDER_PATTERNS = [
    r"(i'd like to|i want to|let's|can we|i will|order) (order|run|get|perform|obtain) (a |an )?(stat )?(12-lead )?(ecg|ekg|chest x-ray|cxr|blood test|labs|troponin|ct scan|ultrasound)",
]

EMPATHY_KEYWORDS = [
    "sorry", "concern", "take care", "help you", "comfortable",
    "breathe", "rest", "right here", "stay calm", "don't worry",
    "take your time", "here for you", "make you comfortable",
    "must be frightening", "understand", "we are going to take care",
    "i hear you", "you are safe", "we'll figure this out", "in good hands",
    "take good care",
]

GENUINELY_UNCLEAR_PATTERNS = [
    r"^how was it\??$",
    r"^what about that\??$",
    r"^and then\??$",
    r"^why\??$",
    r"^what\??$",
    r"^[a-z]{1,4}$",
]


class QuestionClassifier:

    @staticmethod
    def classify(query_text: str) -> ClassifiedIntent:
        query = query_text.lower().strip()
        tokens = query.split()

        # 1. Hidden Diagnosis Request (checked first for clear diagnosis shield)
        if any(re.search(pat, query) for pat in DIAGNOSIS_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.DIAGNOSIS_REQUEST,
                subconcept="diagnosis_shield",
                ui_category="General"
            )

        # 2. Prompt Injection Shield
        if any(re.search(pat, query) for pat in PROMPT_INJECTION_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PROMPT_INJECTION,
                ui_category="Security"
            )

        # 3. Hidden Investigation Result Shield
        if any(re.search(pat, query) for pat in INVESTIGATION_RESULT_SHIELD_PATTERNS):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.INVESTIGATION_REQUEST,
                subconcept="investigation_result_shield",
                ui_category="Diagnostics"
            )

        # 4. Examination Request
        if any(re.search(pat, query) for pat in EXAM_ACTION_PATTERNS):
            target = "vitals" if any(v in query for v in ["vital", "blood pressure", "pulse", "temp"]) else "physical_exam"
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.EXAMINATION_REQUEST,
                action_target=target,
                ui_category="Exam"
            )

        # 5. Investigation Order Request
        if any(re.search(pat, query) for pat in INVESTIGATION_ORDER_PATTERNS):
            target = "ecg" if "ecg" in query or "ekg" in query else "labs"
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.INVESTIGATION_REQUEST,
                action_target=target,
                ui_category="Diagnostics"
            )

        # 6. Empathy Detection
        empathy_detected = any(phrase in query for phrase in EMPATHY_KEYWORDS)

        # 7. Bedside Reassurance / Pure Empathy Statement
        clinical_keywords = [
            "pain", "when did", "where is", "rate", "scale of 1", "sweat", "short of breath", "nausea", "fever", "history",
            "medicine", "allerg", "vomit", "body ache", "cough", "diarrhea"
        ]
        if empathy_detected and len(tokens) <= 25 and not any(k in query for k in clinical_keywords):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.EMPATHY,
                ui_category="General",
                empathy_detected=True
            )


        # 8. Greetings
        greeting_pattern = r"^(hi+|hello+|hey+|howdy|greetings|good morning|good afternoon|good evening|doctor|dr\b)"
        is_greeting = bool(re.search(greeting_pattern, query)) or query.strip() in ["hi", "hii", "hiii", "hello", "helloo", "hey", "heyy"]
        if is_greeting and len(tokens) <= 6 and not any(k in query for k in ["pain", "start", "what brought"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.GREETING,
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 9. "How are you feeling?" / Small Talk
        if re.search(r"how are you (feeling|doing)|how do you feel today|how are you\b", query) and len(tokens) <= 6:
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SMALL_TALK,
                subconcept="feeling_state",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 10. Open-Ended Chief Complaint ("What brought you in today?", "Why did you come?", etc.)
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

        # 11. Allergies
        if any(k in query for k in ["allerg", "allergic", "drug reaction"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ALLERGIES,
                subconcept="allergies",
                ui_category="Allergies",
                empathy_detected=empathy_detected
            )

        # 12. Associated Symptoms & Review of Systems
        # Fever / Chills
        if any(k in query for k in ["fever", "feverish", "hot and cold", "chills", "temperature", "shiver"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="fever",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Body Pain / Muscle Aches
        if any(k in query for k in ["body pain", "body ache", "body aches", "muscle pain", "muscle aches", "myalgia", "generalized pain", "joint pain", "hurting all over"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="body_aches",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Shortness of Breath / Dyspnea
        if any(k in query for k in ["short of breath", "shortness of breath", "breathless", "breathing", "winded", "dyspnea", "wheez", "catch your breath"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="shortness_of_breath",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Sweating / Diaphoresis
        if any(k in query for k in ["sweat", "sweating", "clammy", "cold sweat", "perspir"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="sweating",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Nausea
        if any(k in query for k in ["nausea", "nauseous", "nausious", "sick to your stomach", "sick to my stomach", "queasy"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="nausea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Vomiting
        if any(k in query for k in ["vomit", "vomiting", "vomating", "vomitng", "throw up", "threw up", "throwing up", "puke", "puking"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="vomiting",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Anorexia / Appetite
        if any(k in query for k in ["appetite", "eating", "eat", "anorexia", "hungry"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="anorexia",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Dizziness / Lightheadedness
        if any(k in query for k in ["dizzy", "dizziness", "lightheaded", "faint", "pass out", "blackout", "unsteady"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="dizziness",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Cough
        if any(k in query for k in ["cough", "coughing", "phlegm", "sputum"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="cough",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Diarrhea / Bowel
        if any(k in query for k in ["diarrhea", "loose stool", "bowel movement", "watery stool"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="diarrhea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Headache
        if any(k in query for k in ["headache", "head ache", "head pain", "migraine"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="headache",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Urinary
        if any(k in query for k in ["urinary", "urinating", "pain when you pee", "burning when you pee", "dysuria", "blood in urine"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ASSOCIATED_SYMPTOM,
                subconcept="urinary",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 13. OPQRST Dimensions
        # Activity at onset
        activity_triggers = [
            r"what were you doing",
            r"what was happening when",
            r"where were you when",
            r"what brought (this|it) on",
            r"activity (at|when)",
            r"when it first came on",
        ]
        if any(re.search(pat, query) for pat in activity_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ONSET_ACTIVITY,
                subconcept="onset_activity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Timing / Onset
        onset_triggers = [
            r"when did (this|the|it|your|this problem|the pain|this pain|this breathing) start",
            r"when did (it|this) begin",
            r"how long ago",
            r"how long has (this|it|the pain|this problem) (been going on|lasted)",
            r"how many (minutes|hours|days)",
            r"time of onset",
        ]
        if any(re.search(pat, query) for pat in onset_triggers) or (
            any(k in query for k in ["when did", "how long ago", "onset", "began"])
            and not any(k in query for k in ["doing", "continuous", "radiat", "scale", "feel like", "describe", "better", "worse"])
        ):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.ONSET_TIMING,
                subconcept="onset_timing",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Continuity / Timing
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

        # Location
        location_triggers = [
            r"where (is|was|are) (the|your)? (pain|discomfort|pressure|tightness|sensation|stomach pain|stomach|chest pain)",
            r"where exactly",
            r"point to where",
            r"location of (the)? (pain|discomfort|stomach)",
            r"where does it hurt",
            r"where.*(hurt|pain|ache|discomfort)",
        ]
        if any(re.search(pat, query) for pat in location_triggers) and not any(k in query for k in ["radiat", "spread", "move", "go"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.LOCATION,
                subconcept="location",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Radiation (Checked before quality so questions like 'does chest tightness radiate' match radiation)
        radiation_triggers = [
            r"radiat",
            r"spread",
            r"move anywhere",
            r"moved",
            r"move from",
            r"go anywhere",
            r"travel",
            r"to your jaw",
            r"to your arm",
            r"down your arm",
            r"to your arms",
            r"into your jaw",
            r"shoulder",
            r"neck",
            r"back",
            r"groin",
        ]
        if any(re.search(pat, query) for pat in radiation_triggers) and not any(k in query for k in ["tearing", "ripping"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.RADIATION,
                subconcept="radiation",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Severity
        severity_triggers = [
            r"1 to 10",
            r"1-10",
            r"scale of 1",
            r"rate your pain",
            r"rate the pain",
            r"how severe",
            r"severity",
            r"pain score",
            r"out of 10",
            r"out of ten",
            r"how bad (is it|is the pain)",
        ]
        if any(re.search(pat, query) for pat in severity_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SEVERITY,
                subconcept="severity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Character / Quality
        quality_triggers = [
            r"feel like",
            r"describe (the|what|your)? (pain|sensation|stomach)",
            r"sharp or dull",
            r"crushing",
            r"tight",
            r"burning",
            r"character",
            r"quality of",
            r"kind of pain",
            r"type of pain",
        ]
        if any(re.search(pat, query) for pat in quality_triggers):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.CHARACTER,
                subconcept="character",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )


        # Aggravating / Relieving
        if any(k in query for k in ["better", "reliev", "resting help", "takes the pain away"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.RELIEVING_FACTORS,
                subconcept="relieving_factors",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )
        if any(k in query for k in ["worse", "aggravat", "exertion", "moving around", "trigger", "makes it worse"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.AGGRAVATING_FACTORS,
                subconcept="aggravating_factors",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # 14. Past Medical History
        if any(k in query for k in ["past medical", "medical history", "health condition", "medical conditions", "diagnosed before", "stroke before", "hospital before", "surgeries"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.PAST_MEDICAL_HISTORY,
                subconcept="past_medical_history",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        # 15. Medications
        if any(k in query for k in ["medicat", "medicine", "pill", "prescription", "inhaler", "taking daily", "what do you take"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.MEDICATIONS,
                subconcept="medications",
                ui_category="Meds",
                empathy_detected=empathy_detected
            )

        # 16. Family History
        if any(k in query for k in ["family", "father", "mother", "parent", "genetic", "heart disease in your family", "heart problems", "runs in your family"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.FAMILY_HISTORY,
                subconcept="family_history",
                ui_category="FamilyHx",
                empathy_detected=empathy_detected
            )

        # 17. Social History
        if any(k in query for k in ["smoke", "tobacco", "cigarette", "smoking", "vape", "vaping", "alcohol", "drink", "wine", "beer", "work", "job", "occupation", "stress", "drugs", "cocaine"]):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.SOCIAL_HISTORY,
                subconcept="social_history",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )

        # 18. Genuinely Unclear Check
        if any(re.fullmatch(pat, query) for pat in GENUINELY_UNCLEAR_PATTERNS) or re.fullmatch(r"[^a-zA-Z0-9\s]+", query):
            return ClassifiedIntent(
                raw_query=query_text,
                category=IntentCategory.UNCLEAR,
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # Fallback to Unclassified Clinical Query
        return ClassifiedIntent(
            raw_query=query_text,
            category=IntentCategory.ASSOCIATED_SYMPTOM,
            subconcept="unclassified_symptom",
            ui_category="General",
            empathy_detected=empathy_detected
        )
