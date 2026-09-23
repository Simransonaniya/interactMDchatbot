"""
InteractMD — Learner Question Analyzer.
9-Category Question Classification & Expanded Semantic Clinical Concept Mapping.
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List


class QuestionCategory(str, Enum):
    CASE_FACT_AVAILABLE = "CASE_FACT_AVAILABLE"
    CASE_FACT_AVAILABLE_AND_NEGATIVE = "CASE_FACT_AVAILABLE_AND_NEGATIVE"
    CASE_FACT_NOT_DOCUMENTED = "CASE_FACT_NOT_DOCUMENTED"
    QUESTION_UNCLEAR = "QUESTION_UNCLEAR"
    NON_MEDICAL_OR_SMALL_TALK = "NON_MEDICAL_OR_SMALL_TALK"
    EXAMINATION_REQUEST = "EXAMINATION_REQUEST"
    INVESTIGATION_REQUEST = "INVESTIGATION_REQUEST"
    DIAGNOSIS_MANAGEMENT_REQUEST = "DIAGNOSIS_MANAGEMENT_REQUEST"
    PROMPT_INJECTION_OR_INTERNAL_DATA_REQUEST = "PROMPT_INJECTION_OR_INTERNAL_DATA_REQUEST"


class FactState(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous |your )?instructions",
    r"show (me )?(the )?(complete )?(case )?json",
    r"system prompt",
    r"developer mode",
    r"case data",
    r"hidden evaluation",
    r"scoring rubric",
    r"dump (the )?case",
    r"override rules",
]

HIDDEN_INVESTIGATION_RESULT_PATTERNS = [
    r"what did (my |the )?(stat )?(12-lead )?ecg (show|say)",
    r"what does the ecg show",
    r"what did (my |the )?blood(work| test|s)? (show|say)",
    r"what (is|are) my (troponin|labs|lab results|enzymes)",
    r"what did (my |the )?(chest )?(x-ray|cxr|ct scan|ultrasound) (show|say)",
    r"result of (the )?(ecg|labs|troponin|imaging)",
]

DIAGNOSIS_PATTERNS = [
    r"what is (your|my) diagnosis",
    r"what (condition|disease) do (i|you) have",
    r"tell me (what )?the diagnosis (is)?",
    r"what do you think is wrong with (me|you)",
]

EXAM_ACTION_PATTERNS = [
    r"(i'd like to|i want to|let me|can i|i will) (check|take|examine|listen to|perform|do) (your )?(vital signs|vitals|blood pressure|heart rate|pulse|temp|temperature)",
    r"(i'd like to|i want to|let me|can i|i will) (perform|do) (a |an )?(cardiovascular|respiratory|chest|abdominal|physical|neuro) (exam|examination)",
    r"(i'd like to|i want to|let me|can i|i will) (listen to|auscultate) (your )?(heart|lungs|chest|breathing|abdomen)",
]

INVESTIGATION_ORDER_PATTERNS = [
    r"(i'd like to|i want to|let's|can we|i will|order) (order|run|get|perform|obtain) (a |an )?(stat )?(12-lead )?(ecg|ekg|chest x-ray|cxr|blood test|labs|troponin|ct scan|ultrasound)",
]

MEDICAL_JARGON_TERMS = [
    "stemi", "myocardial infarction", "troponin", "ecg", "ekg", "ischemia",
    "appendicitis", "peritonitis", "bronchospasm", "pneumonia", "sepsis",
    "curb-65", "mcburney", "rovsing", "psoas", "egophony", "diaphoresis",
    "tachycardia", "bradycardia", "nstemi", "echocardiogram", "leukocytosis",
    "bandemia", "pathognomonic", "atherothrombotic", "pericarditis", "aortic dissection",
]

EMPATHY_PHRASES = [
    "sorry", "concern", "take care", "help you", "comfortable",
    "breathe", "rest", "right here", "stay calm", "don't worry",
    "take your time", "here for you", "make you comfortable",
    "must be frightening", "understand", "we are going to take care",
    "i hear you", "you are safe", "we'll figure this out", "in good hands",
]

GENUINELY_UNCLEAR_PATTERNS = [
    r"^how was it\??$",
    r"^what about that\??$",
    r"^and then\??$",
    r"^why\??$",
    r"^what\??$",
    r"^[a-z]{1,4}$", # single short random letters e.g. "jhjh", "asdf"
]


class AnalyzedQuestion:
    def __init__(
        self,
        raw_text: str,
        category: QuestionCategory,
        clinical_concept: Optional[str] = None,
        ui_category: str = "General",
        empathy_detected: bool = False,
        jargon_term: Optional[str] = None,
        action_target: Optional[str] = None,
        is_genuinely_unclear: bool = False
    ):
        self.raw_text = raw_text
        self.category = category
        self.clinical_concept = clinical_concept
        self.ui_category = ui_category
        self.empathy_detected = empathy_detected
        self.jargon_term = jargon_term
        self.action_target = action_target
        self.is_genuinely_unclear = is_genuinely_unclear

    def __repr__(self):
        return f"<AnalyzedQuestion cat={self.category.value} concept={self.clinical_concept} ui_cat={self.ui_category}>"


class LearnerQuestionAnalyzer:

    @staticmethod
    def analyze(query_text: str) -> AnalyzedQuestion:
        query = query_text.lower().strip()
        tokens = query.split()

        # 1. Prompt Injection / Internal Data Request
        if any(re.search(p, query) for p in PROMPT_INJECTION_PATTERNS):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.PROMPT_INJECTION_OR_INTERNAL_DATA_REQUEST,
                ui_category="General"
            )

        # 2. Hidden Investigation Result Request
        if any(re.search(p, query) for p in HIDDEN_INVESTIGATION_RESULT_PATTERNS):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.INVESTIGATION_REQUEST,
                clinical_concept="investigation_result_shield",
                ui_category="General"
            )

        # 3. Hidden Diagnosis / Management Request
        if any(re.search(p, query) for p in DIAGNOSIS_PATTERNS):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.DIAGNOSIS_MANAGEMENT_REQUEST,
                clinical_concept="diagnosis_shield",
                ui_category="General"
            )

        # 4. Examination Action Request
        if any(re.search(p, query) for p in EXAM_ACTION_PATTERNS):
            target = "vitals" if any(v in query for v in ["vital", "blood pressure", "pulse", "temp"]) else "exam"
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.EXAMINATION_REQUEST,
                action_target=target,
                ui_category="Exam"
            )

        # 5. Investigation Order Request
        if any(re.search(p, query) for p in INVESTIGATION_ORDER_PATTERNS):
            target = "ecg" if "ecg" in query or "ekg" in query else "labs"
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.INVESTIGATION_REQUEST,
                action_target=target,
                ui_category="Diagnostics"
            )

        # 6. Empathy Detection
        empathy_detected = any(phrase in query for phrase in EMPATHY_PHRASES)

        # 7. Medical Jargon Layperson Clarification
        used_jargon = [t for t in MEDICAL_JARGON_TERMS if re.search(r'\b' + re.escape(t) + r'\b', query)]
        if used_jargon and len(tokens) <= 10 and not any(k in query for k in ["when", "how", "pain", "where", "history", "medicine"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="medical_jargon",
                jargon_term=used_jargon[0].upper(),
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 8. Bedside Reassurance / Empathy Only
        clinical_keywords = [
            "pain", "when", "where", "how", "feel", "start", "radiat", "spread",
            "scale", "rate", "sweat", "short of breath", "trouble breathing", "nausea", "fever", "history",
            "medicine", "smoke", "drink", "family", "allerg", "cough", "vomit",
            "doing", "continuous", "better", "worse", "body", "ache", "cold", "diarrhea", "headache"
        ]
        if empathy_detected and len(tokens) < 18 and not any(k in query for k in clinical_keywords):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="bedside_empathy",
                ui_category="General",
                empathy_detected=True
            )

        # 9. Greetings
        if re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening|doctor)\b", query) and len(tokens) <= 5:
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="greeting",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 10. Closings / Leaving / Stepping Out / Goodbyes
        closing_patterns = [
            r"^(bye|goodbye|bye doctor|see you|see you later|cya|take care)\b",
            r"(be right back|i'll be right back|i will be back|step out|stepping out|give me a moment|let me check on that|check your results|talk to the attending)",
        ]
        if any(re.search(pat, query) for pat in closing_patterns):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="closing",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 11. Polite Acknowledgments / Confirmations ("ok", "okay", "got it", "i see", "understood", "sure", "alright")
        ack_words = ["ok", "okay", "alright", "all right", "got it", "i see", "understood", "sure", "sounds good", "noted", "cool", "perfect", "yes", "yeah"]
        if query.rstrip(".!?") in ack_words or (len(tokens) <= 3 and any(query.startswith(w) for w in ["okay", "ok", "got it", "understood", "alright"])):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="polite_ack",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 12. Gratitude / Thanks
        if any(w in query for w in ["thank you", "thanks", "appreciate it", "thank you doctor", "thanks doctor"]) and len(tokens) <= 6:
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="gratitude",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 13. "How are you feeling?" Small Talk
        if re.search(r"how are you (feeling|doing)|how do you feel today|how are you", query) and len(tokens) <= 6:
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                clinical_concept="small_talk_feeling",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 14. Open-Ended Chief Complaint
        open_ended_triggers = [
            r"what (brought|brings) you",
            r"how can i help",
            r"tell me (about |what )?(what happened|what's going on|what brings you)",
            r"what('s| is) (the matter|going on|wrong|troubling you|the problem)",
            r"what happened",
            r"why (are you|did you come|did you call)",
            r"chief complaint",
        ]
        if any(re.search(pat, query) for pat in open_ended_triggers):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="chief_complaint",
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # 15. Allergies (Checked before meds to catch "allergies to medications")
        if any(k in query for k in ["allerg", "allergic", "drug reaction"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="allergies",
                ui_category="Allergies",
                empathy_detected=empathy_detected
            )

        # ------------------------------------------------------------------
        # REVIEW OF SYSTEMS & ASSOCIATED SYMPTOMS (With Tri-State Support)
        # ------------------------------------------------------------------
        
        # Fever / Chills
        if any(k in query for k in ["fever", "feverish", "hot and cold", "chills", "temperature", "shiver", "did you feel fever", "any fever"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="fever",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Body Pain / Muscle Aches / Body Aches
        if any(k in query for k in ["body pain", "body ache", "body aches", "muscle pain", "muscle aches", "myalgia", "generalized pain", "joint pain", "hurting all over"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                clinical_concept="body_aches",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Cough
        if any(k in query for k in ["cough", "coughing", "hacking", "productive cough", "dry cough"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="cough",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Cold / Upper Respiratory
        if any(k in query for k in ["cold", "runny nose", "congestion", "sore throat", "stuffy nose", "sneezing", "upper respiratory"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="cold_symptoms",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Vomiting / Emesis
        if any(k in query for k in ["vomit", "vomiting", "throw up", "threw up", "throwing up"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="vomiting",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Nausea
        if any(k in query for k in ["nausea", "nauseous", "sick to your stomach", "queasy"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="nausea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Diarrhea / Bowel
        if any(k in query for k in ["diarrhea", "loose stool", "loose stools", "bowel movement", "watery stool"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                clinical_concept="diarrhea",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Headache / Head Pain
        if any(k in query for k in ["headache", "head ache", "head pain", "migraine"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                clinical_concept="headache",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Urinary Symptoms
        if any(k in query for k in ["urinary", "urinating", "pain when you pee", "burning when you pee", "dysuria", "blood in urine"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                clinical_concept="urinary",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Shortness of Breath / Dyspnea
        if any(k in query for k in ["short of breath", "shortness of breath", "breathless", "breathing", "winded", "dyspnea", "wheez"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="shortness_of_breath",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Sweating / Diaphoresis
        if any(k in query for k in ["sweat", "sweating", "clammy", "cold sweat", "perspir"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="sweating",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Dizziness / Lightheadedness
        if any(k in query for k in ["dizzy", "dizziness", "lightheaded", "faint", "pass out", "blackout", "unsteady"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="dizziness",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Tearing / Ripping Pain
        if any(k in query for k in ["tearing", "ripping", "between your shoulder blades", "shoulder blades"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="tearing_pain",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # ------------------------------------------------------------------
        # OPQRST / HPI CHARACTERISTICS
        # ------------------------------------------------------------------

        # Activity Context at Onset
        activity_triggers = [
            r"what were you doing",
            r"what was happening when",
            r"where were you when",
            r"what brought (this|it) on",
            r"activity (at|when)",
            r"when it first came on",
        ]
        if any(re.search(pat, query) for pat in activity_triggers):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="onset_activity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Timing / Duration of Onset
        onset_triggers = [
            r"when did (this|the|it|your|this problem|the pain|this pain|this breathing) start",
            r"when did (it|this) begin",
            r"how long ago",
            r"how long has (this|it|the pain|this problem) (been going on|lasted)",
            r"how many (minutes|hours|days)",
            r"time of onset",
        ]
        if any(re.search(pat, query) for pat in onset_triggers) or (
            any(k in query for k in ["when did", "how long ago", "start", "onset", "began"])
            and not any(k in query for k in ["doing", "continuous", "radiat", "scale", "feel like", "describe", "better", "worse"])
        ):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="onset_timing",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Continuity
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
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="continuity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Location
        location_triggers = [
            r"where (is|was|are) (the|your)? (pain|discomfort|pressure|tightness|sensation)",
            r"where exactly",
            r"point to where",
            r"location of (the)? (pain|discomfort)",
        ]
        if any(re.search(pat, query) for pat in location_triggers) and not any(k in query for k in ["radiat", "spread", "move", "go"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="location",
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
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="severity",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Quality / Character
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
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="character",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Radiation
        radiation_triggers = [
            r"radiat",
            r"spread",
            r"move anywhere",
            r"go anywhere",
            r"to your jaw",
            r"to your arm",
            r"jaw",
            r"arm",
            r"shoulder",
            r"neck",
            r"back",
            r"groin",
            r"travel",
        ]
        if any(re.search(pat, query) for pat in radiation_triggers) and not any(k in query for k in ["tearing", "ripping"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="radiation",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Aggravating / Relieving
        if any(k in query for k in ["better", "reliev", "resting help"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="relieving",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )
        if any(k in query for k in ["worse", "aggravat", "exertion", "moving around", "trigger", "makes it worse"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="aggravating",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Prior Episodes
        prior_triggers = [
            r"happened before",
            r"had this (pain|problem|before)",
            r"felt this before",
            r"ever had a heart attack",
            r"prior episode",
            r"first time",
        ]
        if any(re.search(pat, query) for pat in prior_triggers):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="prior_episodes",
                ui_category="HPI",
                empathy_detected=empathy_detected
            )

        # Past Medical History
        if any(k in query for k in ["past medical", "medical history", "health condition", "medical conditions", "diagnosed before", "stroke before", "hospital before", "surgeries"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="past_medical_history",
                ui_category="PMH",
                empathy_detected=empathy_detected
            )

        # Medications
        if any(k in query for k in ["medicat", "medicine", "pill", "prescription", "inhaler", "taking daily", "what do you take"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="medications",
                ui_category="Meds",
                empathy_detected=empathy_detected
            )

        # Family History
        if any(k in query for k in ["family", "father", "mother", "parent", "genetic", "heart disease in your family", "heart problems", "runs in your family"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="family_history",
                ui_category="FamilyHx",
                empathy_detected=empathy_detected
            )

        # Social History
        if any(k in query for k in ["smoke", "tobacco", "cigarette", "smoking", "vape", "vaping"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="social_smoking",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )
        if any(k in query for k in ["alcohol", "drink", "wine", "beer"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="social_alcohol",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )
        if any(k in query for k in ["work", "job", "occupation", "stress", "living", "drugs", "cocaine"]):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.CASE_FACT_AVAILABLE,
                clinical_concept="social_general",
                ui_category="SocialHx",
                empathy_detected=empathy_detected
            )

        # ------------------------------------------------------------------
        # GENUINELY UNCLEAR / GARBLED / AMBIGUOUS QUESTION CHECK
        # ------------------------------------------------------------------
        if any(re.fullmatch(pat, query) for pat in GENUINELY_UNCLEAR_PATTERNS) or re.fullmatch(r"[^a-zA-Z0-9\s]+", query):
            return AnalyzedQuestion(
                raw_text=query_text,
                category=QuestionCategory.QUESTION_UNCLEAR,
                is_genuinely_unclear=True,
                ui_category="General",
                empathy_detected=empathy_detected
            )

        # Unclassified symptom inquiry (Case Fact Not Documented)
        return AnalyzedQuestion(
            raw_text=query_text,
            category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
            clinical_concept="unclassified_symptom",
            ui_category="General",
            empathy_detected=empathy_detected
        )
