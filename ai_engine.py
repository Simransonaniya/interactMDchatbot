"""
AI Patient Dialogue Engine & Clinical Guardrails.

Enforces the following Patient Dialogue Rules:
  - Always speak as the patient in first person.
  - Never refer to yourself as "he", "she", or "the patient".
  - Never expose raw case JSON or case wording directly.
  - Convert structured clinical facts into natural patient language.
  - Only reveal information relevant to the learner's current question.
  - Do not volunteer unrelated symptoms, diagnosis, investigation results,
    or hidden case facts.
  - Do not summarize the entire case in one response.
  - Respond like a real patient, not like a medical assistant.
  - Keep answers concise and conversational.
  - Preserve consistency with previously revealed information.
  - Open-ended questions → chief complaint only; wait for follow-ups.
"""

import os
import re
from typing import Dict, Any, List, Set, Optional

# ---------------------------------------------------------------------------
# Keyword bank: Empathy phrases the student might say
# ---------------------------------------------------------------------------
EMPATHY_PHRASES = [
    "sorry", "concern", "take care", "help you", "comfortable",
    "breathe", "rest", "right here", "stay calm", "don't worry",
    "take your time", "here for you", "make you comfortable",
    "must be frightening", "understand", "we are going to take care",
    "i hear you", "you are safe", "we'll figure this out",
]

# ---------------------------------------------------------------------------
# Medical jargon that a layperson patient would not understand
# ---------------------------------------------------------------------------
MEDICAL_JARGON_TERMS = [
    "stemi", "myocardial infarction", "troponin", "ecg", "ischemia",
    "appendicitis", "peritonitis", "bronchospasm", "pneumonia",
    "sepsis", "curb-65", "mcburney", "rovsing", "psoas", "egophony",
    "diaphoresis", "tachycardia", "bradycardia", "nstemi", "echocardiogram",
    "leukocytosis", "bandemia", "pathognomonic", "atherothrombotic",
]

# ---------------------------------------------------------------------------
# Open-ended / chief-complaint-style question triggers
# ---------------------------------------------------------------------------
OPEN_ENDED_TRIGGERS = [
    r"what brings you",
    r"how can i help",
    r"tell me about",
    r"what('s| is) (the matter|going on|wrong|bothering|troubling)",
    r"what happened",
    r"how are you (feeling|doing)",
    r"what (can i|may i) (do|help)",
    r"why (are you|did you come|did you call)",
    r"chief complaint",
    r"present(ing)? complaint",
    r"what seems to be",
]

# ---------------------------------------------------------------------------
# Category labels for the disclosed-facts tracker
# ---------------------------------------------------------------------------
CATEGORY_ONSET      = "onset"
CATEGORY_QUALITY    = "quality"
CATEGORY_RADIATION  = "radiation"
CATEGORY_SEVERITY   = "severity"
CATEGORY_PALLIATION = "palliation"
CATEGORY_SWEAT      = "sweat"
CATEGORY_NAUSEA     = "nausea"
CATEGORY_RESP       = "respiratory"
CATEGORY_FEVER      = "fever"
CATEGORY_BACK_PAIN  = "back_pain"
CATEGORY_PMH        = "pmh"
CATEGORY_MEDS       = "medications"
CATEGORY_ALLERGIES  = "allergies"
CATEGORY_FAMILY_HX  = "family_hx"
CATEGORY_SOCIAL_HX  = "social_hx"
CATEGORY_EMPATHY    = "empathy"
CATEGORY_GREETING   = "greeting"
CATEGORY_CHIEF      = "chief_complaint"


class AIPatientEngine:

    def __init__(self):
        self.provider_name = os.getenv(
            "LLM_PROVIDER", "InteractMD Python AI Patient Core"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process_turn(
        self,
        case_data: Dict[str, Any],
        user_message: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Processes a doctor-student message and returns realistic patient dialogue
        strictly using the provided case_data from the database.

        All replies comply with the Patient Dialogue Rules:
          - First-person only
          - No raw JSON facts exposed verbatim
          - No 3rd-person pronouns
          - Progressive disclosure (only what was asked)
        """
        query  = user_message.lower().strip()
        facts  = case_data.get("facts", {})
        patient = case_data.get("patient", {})

        # ------------------------------------------------------------------
        # 0. Build the set of categories already disclosed this session
        # ------------------------------------------------------------------
        disclosed: Set[str] = self._disclosed_categories(conversation_history)

        # ------------------------------------------------------------------
        # 1. Detect bedside empathy in the student's message
        # ------------------------------------------------------------------
        empathy_detected = any(phrase in query for phrase in EMPATHY_PHRASES)

        # ------------------------------------------------------------------
        # 2. Check for medical jargon directed at the patient
        # ------------------------------------------------------------------
        used_jargon = [t for t in MEDICAL_JARGON_TERMS if t in query]
        if (
            used_jargon
            and len(query.split()) < 15
            and not any(k in query for k in ["when", "how", "pain", "where"])
        ):
            jargon_term = used_jargon[0].upper()
            return self._build(
                reply=(
                    f"I... I don't know what {jargon_term} means, doctor... "
                    "Is that something serious? Please, just tell me what's happening to me."
                ),
                empathy_detected=empathy_detected,
                category="General",
                suggested_topics=["Explain in plain language", "Ask about symptoms", "Reassure patient"],
            )

        # ------------------------------------------------------------------
        # 3. Greeting (student opens with hello / good morning / etc.)
        # ------------------------------------------------------------------
        if (
            re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening|doctor)\b", query)
            and len(query) < 40
        ):
            disclosed.add(CATEGORY_GREETING)
            presentation = patient.get("presentationComplaint") or facts.get("chiefComplaint") or "I am really not feeling well."
            return self._build(
                reply=f"Hello, doctor... Thank you for seeing me. {presentation}",
                empathy_detected=empathy_detected,
                category="General",
                suggested_topics=["Onset & Timing", "Location & Quality", "Severity"],
            )

        # ------------------------------------------------------------------
        # 4. Open-ended question → chief complaint only, not full history
        # ------------------------------------------------------------------
        if any(re.search(pattern, query) for pattern in OPEN_ENDED_TRIGGERS):
            presentation = patient.get("presentationComplaint") or facts.get("chiefComplaint") or "I've been in a lot of discomfort."
            return self._build(
                reply=(
                    f"{presentation} "
                    "I've been feeling quite concerned, to be honest."
                ),
                empathy_detected=empathy_detected,
                category="General",
                suggested_topics=["When did it start?", "Where exactly is the pain?", "Describe the pain"],
            )

        # ------------------------------------------------------------------
        # 5. Pure empathy / comfort (no clinical question embedded)
        # ------------------------------------------------------------------
        if (
            empathy_detected
            and len(query) < 55
            and not any(k in query for k in ["pain", "when", "where", "history", "medicine", "smoke", "drink", "fever", "nausea"])
        ):
            return self._build(
                reply=(
                    "Thank you so much, doctor. That genuinely gives me some comfort... "
                    "I just want to figure out what's causing this."
                ),
                empathy_detected=True,
                category="General",
                suggested_topics=["Symptom onset", "Describe pain", "Medical history"],
            )

        # ------------------------------------------------------------------
        # 6. OPQRST — Onset & Timing
        # ------------------------------------------------------------------
        if any(k in query for k in ["when", "start", "how long", "onset", "began", "time", "duration", "hours", "days"]):
            onset_raw   = self._personalize(facts.get("onset", "it came on quite suddenly."))
            timing_raw  = self._personalize(facts.get("timing", "it's been continuous since then."))
            if CATEGORY_ONSET in disclosed:
                return self._build(
                    reply=f"As I mentioned, it {onset_raw} And {timing_raw}",
                    empathy_detected=empathy_detected,
                    category="HPI",
                    suggested_topics=["Radiation", "Severity", "Provocation"],
                )
            return self._build(
                reply=f"It {onset_raw} And since then, {timing_raw}",
                empathy_detected=empathy_detected,
                category="HPI",
                suggested_topics=["Radiation", "Severity", "Provocation"],
                disclose=CATEGORY_ONSET,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 7. OPQRST — Provocation & Palliation
        # ------------------------------------------------------------------
        if any(k in query for k in ["better", "worse", "aggravat", "reliev", "trigger", "rest", "moving", "cough", "position"]):
            pp = self._personalize(facts.get("provocationPalliative", "nothing seems to make it noticeably better or worse."))
            return self._build(
                reply=f"Honestly, {pp}",
                empathy_detected=empathy_detected,
                category="HPI",
                suggested_topics=["Quality of pain", "Associated symptoms"],
                disclose=CATEGORY_PALLIATION,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 8. OPQRST — Quality & Nature
        # ------------------------------------------------------------------
        if any(k in query for k in ["feel like", "describe", "sharp", "dull", "crushing", "tight", "burning", "nature", "kind of pain", "type of pain", "character", "sensation"]):
            quality = self._personalize(facts.get("quality", "an uncomfortable, aching pressure."))
            return self._build(
                reply=f"It feels like {quality}",
                empathy_detected=empathy_detected,
                category="HPI",
                suggested_topics=["Radiation", "Severity (1–10)"],
                disclose=CATEGORY_QUALITY,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 9. OPQRST — Radiation / Spread
        # ------------------------------------------------------------------
        if any(k in query for k in ["radiat", "spread", "move", "go anywhere", "jaw", "arm", "back", "neck", "shoulder", "travel", "groin", "flank"]):
            rad_raw = facts.get("radiation")
            if rad_raw and "no" not in rad_raw.lower():
                reply = f"Yes, {self._personalize(rad_raw)}"
            else:
                reply = "No — it stays right where it is. It hasn't moved or spread anywhere else."
            return self._build(
                reply=reply,
                empathy_detected=empathy_detected,
                category="HPI",
                suggested_topics=["Severity", "Associated symptoms"],
                disclose=CATEGORY_RADIATION,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 10. OPQRST — Severity / Pain Scale
        # ------------------------------------------------------------------
        if any(k in query for k in ["scale", "rate", "how bad", "severity", "1 to 10", "1-10", "score", "out of ten", "out of 10", "pain level"]):
            severity_raw = self._personalize(facts.get("severity", "it's pretty severe, maybe an 8 out of 10."))
            severity_clean = re.sub(r'^\s*(rates?\s+(it\s+)?)', 'I\'d say ', severity_raw, flags=re.IGNORECASE)
            return self._build(
                reply=f"Right now? {severity_clean}",
                empathy_detected=empathy_detected,
                category="HPI",
                suggested_topics=["Past medical history", "Current medications"],
                disclose=CATEGORY_SEVERITY,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 11. Associated — Sweating / Diaphoresis
        # ------------------------------------------------------------------
        if any(k in query for k in ["sweat", "clammy", "perspir", "cold sweat", "damp"]):
            has_sweat = any(
                "sweat" in s.lower() or "diaphoresis" in s.lower()
                for s in facts.get("associatedSymptoms", [])
            )
            reply = (
                "Yes, I'm noticeably sweaty and feeling clammy."
                if has_sweat
                else "No, I haven't noticed any unusual sweating."
            )
            return self._build(
                reply=reply,
                empathy_detected=empathy_detected,
                category="HPI",
                disclose=CATEGORY_SWEAT,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 12. Associated — Nausea & Vomiting
        # ------------------------------------------------------------------
        if any(k in query for k in ["nausea", "nauseous", "throw up", "vomit", "sick to your stomach", "queasy"]):
            n = [
                s for s in facts.get("associatedSymptoms", [])
                if "nausea" in s.lower() or "vomit" in s.lower()
            ]
            reply = (
                f"Yes — {self._personalize(n[0]).rstrip('.')}."
                if n
                else "No, I haven't felt nauseous or vomited."
            )
            return self._build(
                reply=reply,
                empathy_detected=empathy_detected,
                category="HPI",
                disclose=CATEGORY_NAUSEA,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 13. Associated — Shortness of Breath / Wheezing
        # ------------------------------------------------------------------
        if any(k in query for k in ["short of breath", "breath", "wheez", "cough", "air", "breathing"]):
            resp_sym = [
                s for s in facts.get("associatedSymptoms", [])
                if any(w in s.lower() for w in ["breath", "wheez", "cough", "dyspnea"])
            ]
            reply = (
                f"Yes — {self._personalize(resp_sym[0]).rstrip('.')}."
                if resp_sym
                else "My breathing is okay, really."
            )
            return self._build(
                reply=reply,
                empathy_detected=empathy_detected,
                category="HPI",
                disclose=CATEGORY_RESP,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 14. Associated — Fever / Chills
        # ------------------------------------------------------------------
        if any(k in query for k in ["fever", "chills", "temperature", "hot", "cold", "shiver"]):
            fever_sym = [
                s for s in facts.get("associatedSymptoms", [])
                if "fever" in s.lower() or "chills" in s.lower() or "temp" in s.lower()
            ]
            reply = (
                f"Yes — {self._personalize(fever_sym[0]).rstrip('.')}."
                if fever_sym
                else "No — I haven't felt feverish or had chills."
            )
            return self._build(
                reply=reply,
                empathy_detected=empathy_detected,
                category="HPI",
                disclose=CATEGORY_FEVER,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 15. Past Medical History
        # ------------------------------------------------------------------
        if any(k in query for k in ["past medical", "medical history", "conditions", "heart attack before", "diagnosed", "health problems", "medical problems", "surgeries"]):
            raw_pmh = facts.get("pastMedicalHistory", ["No known prior medical problems"])
            if not raw_pmh:
                raw_pmh = ["No known significant chronic medical conditions."]
            pmh_items = [self._personalize(item) for item in raw_pmh]
            if len(pmh_items) == 1:
                pmh_text = pmh_items[0]
            else:
                pmh_text = ", ".join(pmh_items[:-1]) + f", and {pmh_items[-1]}"
            return self._build(
                reply=f"Well, I've had {pmh_text}.",
                empathy_detected=empathy_detected,
                category="PMH",
                suggested_topics=["Current medications", "Allergies", "Family history"],
                disclose=CATEGORY_PMH,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 16. Current Medications
        # ------------------------------------------------------------------
        if any(k in query for k in ["medicat", "medicine", "pill", "prescription", "inhaler", "taking", "drugs"]):
            raw_meds = facts.get("medications", ["No regular medications"])
            if not raw_meds:
                raw_meds = ["I don't take any regular medications."]
            meds_items = [self._personalize(m) for m in raw_meds]
            if len(meds_items) == 1:
                meds_text = meds_items[0]
            else:
                meds_text = ", ".join(meds_items[:-1]) + f", and {meds_items[-1]}"
            return self._build(
                reply=f"I take {meds_text}.",
                empathy_detected=empathy_detected,
                category="Meds",
                suggested_topics=["Allergies", "Family history"],
                disclose=CATEGORY_MEDS,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 17. Allergies
        # ------------------------------------------------------------------
        if "allerg" in query:
            raw_allergies = facts.get("allergies", ["No known allergies"])
            if not raw_allergies:
                raw_allergies = ["No known drug allergies."]
            allergy_text = ", ".join(self._personalize(a) for a in raw_allergies)
            return self._build(
                reply=f"As far as I know — {allergy_text}.",
                empathy_detected=empathy_detected,
                category="Allergies",
                suggested_topics=["Family history", "Social habits"],
                disclose=CATEGORY_ALLERGIES,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 18. Family History
        # ------------------------------------------------------------------
        if any(k in query for k in ["family", "father", "mother", "parents", "genetics", "hereditary", "runs in"]):
            raw_fh = facts.get("familyHistory", ["No significant family history"])
            if isinstance(raw_fh, list):
                raw_fh = " ".join(raw_fh)
            fh_text = self._personalize(raw_fh)
            return self._build(
                reply=f"In terms of my family — {fh_text}",
                empathy_detected=empathy_detected,
                category="FamilyHx",
                suggested_topics=["Smoking / Alcohol", "Physical examination"],
                disclose=CATEGORY_FAMILY_HX,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 19. Social History — Smoking, Alcohol, Work, Stress
        # ------------------------------------------------------------------
        if any(k in query for k in ["smoke", "tobacco", "cigarette", "alcohol", "drink", "wine", "beer", "drug", "work", "stress", "job", "occupation"]):
            raw_sh = facts.get("socialHistory", "Non-smoker. I drink occasionally.")
            if isinstance(raw_sh, list):
                raw_sh = " ".join(raw_sh)
            sh_text = self._personalize(raw_sh)
            return self._build(
                reply=f"Regarding my lifestyle — {sh_text}",
                empathy_detected=empathy_detected,
                category="SocialHx",
                suggested_topics=["Physical examination", "Diagnostic tests"],
                disclose=CATEGORY_SOCIAL_HX,
                disclosed=disclosed,
            )

        # ------------------------------------------------------------------
        # 20. Default / Fallback — Realistic patient clarification
        # ------------------------------------------------------------------
        return self._build(
            reply=(
                "I'm not completely sure about that, doctor... "
                f"What I know is that {facts.get('chiefComplaint', patient.get('presentationComplaint', 'I am really not feeling well'))}. "
                "Is there anything else you need me to explain?"
            ),
            empathy_detected=empathy_detected,
            category="General",
            suggested_topics=["Onset & Timing", "Location & Character", "Medical History"],
        )

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _disclosed_categories(self, history: List[Dict[str, Any]]) -> Set[str]:
        """Scans conversation history metadata to determine already-disclosed topics."""
        disclosed = set()
        for turn in history:
            cat = turn.get("category")
            if cat:
                disclosed.add(cat.lower())
        return disclosed

    def _personalize(self, text: Any) -> str:
        """Converts objective 3rd-person clinical strings into 1st-person patient language."""
        if not text:
            return ""
        if isinstance(text, list):
            text = ", ".join(str(t) for t in text)
        s = str(text).strip()

        # Pronoun conversions
        s = re.sub(r"\bhis\b", "my", s, flags=re.IGNORECASE)
        s = re.sub(r"\bher\b", "my", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhim\b", "me", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhe is\b", "I am", s, flags=re.IGNORECASE)
        s = re.sub(r"\bshe is\b", "I am", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhe has\b", "I have", s, flags=re.IGNORECASE)
        s = re.sub(r"\bshe has\b", "I have", s, flags=re.IGNORECASE)
        s = re.sub(r"\bpatient reports\b", "I noticed", s, flags=re.IGNORECASE)
        s = re.sub(r"\bpatient states\b", "I felt", s, flags=re.IGNORECASE)
        s = re.sub(r"\bpatient\b", "I", s, flags=re.IGNORECASE)
        return s

    def _build(
        self,
        reply: str,
        empathy_detected: bool,
        category: str,
        suggested_topics: Optional[List[str]] = None,
        disclose: Optional[str] = None,
        disclosed: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if disclose and disclosed is not None:
            disclosed.add(disclose)
        return {
            "reply": reply,
            "empathy_detected": empathy_detected,
            "category": category,
            "provider": self.provider_name,
            "suggested_topics": suggested_topics or [],
        }
