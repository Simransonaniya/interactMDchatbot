"""
InteractMD — Patient Response Validator & Sanitizer.
Ensures first-person patient voice, blocks hidden diagnostic leaks, enforces length constraints,
detects contradictions with active session state, and shields system prompts.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from patient_state import PatientSimulationState


THIRD_PERSON_VIOLATIONS = [
    (r"\bthe patient\b", "I"),
    (r"\bthe patient's\b", "my"),
    (r"\bhis chest\b", "my chest"),
    (r"\bher chest\b", "my chest"),
    (r"\bhis pain\b", "my pain"),
    (r"\bher pain\b", "my pain"),
    (r"\bhis arm\b", "my arm"),
    (r"\bher arm\b", "my arm"),
    (r"\bhis jaw\b", "my jaw"),
    (r"\bher jaw\b", "my jaw"),
    (r"\bhis symptoms\b", "my symptoms"),
    (r"\bher symptoms\b", "my symptoms"),
    (r"\bthe case\b", "what's happening"),
    (r"\bthe subject\b", "I"),
    (r"\bpatient reports\b", "I noticed"),
    (r"\bpatient states\b", "I felt"),
    (r"\bpatient presents with\b", "I started having"),
    (r"\brates it\b", "I would rate it"),
    (r"\bdenies\b", "I don't have"),
    (r"\bnkda\b", "no known drug allergies"),
]

PRONOUN_LEAKS = [
    (r"\bhe has\b", "I have"),
    (r"\bshe has\b", "I have"),
    (r"\bhe is\b", "I am"),
    (r"\bshe is\b", "I am"),
    (r"\bhe was\b", "I was"),
    (r"\bshe was\b", "I was"),
    (r"\bhe felt\b", "I felt"),
    (r"\bshe felt\b", "I felt"),
    (r"\bhe started\b", "I started"),
    (r"\bshe started\b", "I started"),
    (r"\bhis\b", "my"),
    (r"\bher\b", "my"),
]

HIDDEN_DIAGNOSIS_TERMS = [
    "stemi", "nstemi", "myocardial infarction", "acute coronary syndrome",
    "acute appendicitis", "peritonitis", "bronchospasm", "pneumonia",
    "leukocytosis", "bandemia", "st elevation", "t-wave inversion", "troponin i",
    "perforated appendix", "sonographic mcburney", "periappendiceal fat"
]

REPETITIVE_DOCTOR_PROMPTS = [
    r"what else do you need to know\??",
    r"is there anything specific you need to check\??",
    r"what would you like to examine next\??",
    r"would you like to ask about my (medical history|medications|allergies)\??",
]

HALLUCINATED_INVENTIONS = [
    r"\bpanic attack\b",
    r"\banxiety attack\b",
    r"\broutine check-ups\b",
    r"\broutine checkup\b",
    r"\bwent out with friends\b",
    r"\bwatched a movie\b",
    r"\bhad dinner and\b",
    r"\bunclassified symptom\b",
    r"\btrouble remembering to take it as prescribed\b",
    r"\bnot really sure if i used it correctly\b",
    r"\bwatching tv\b",
    r"\bwoke up this morning\b",
    r"\bsitting up in bed\b",
]


class ValidationResult:
    def __init__(self, is_valid: bool, sanitized_text: str, reason: Optional[str] = None):
        self.is_valid = is_valid
        self.sanitized_text = sanitized_text
        self.reason = reason


class PatientResponseValidator:

    @staticmethod
    def validate(
        raw_response: Optional[str],
        fallback_statement: str,
        session_state: Optional[PatientSimulationState] = None
    ) -> ValidationResult:
        if not raw_response or not raw_response.strip():
            return ValidationResult(is_valid=False, sanitized_text=fallback_statement, reason="Empty response")

        text = raw_response.strip()

        # 1. Clean markdown / JSON artifacts
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.replace('"', '').replace("'", "'").strip()

        # 2. Block AI Assistant Meta Commentary
        meta_phrases = ["as an ai", "i am an ai", "as a virtual patient", "in this simulation", "as a clinical patient"]
        if any(mp in text.lower() for mp in meta_phrases):
            return ValidationResult(is_valid=False, sanitized_text=fallback_statement, reason="Meta-assistant leak")

        # 3. Block Hidden Medical Diagnosis Leaks
        lower = text.lower()
        for term in HIDDEN_DIAGNOSIS_TERMS:
            if term in lower:
                return ValidationResult(
                    is_valid=False,
                    sanitized_text=fallback_statement,
                    reason=f"Hidden medical diagnosis term leaked: {term}"
                )

        # 4. Clean third-person clinical chart phrasing into direct first-person
        for pattern, replacement in THIRD_PERSON_VIOLATIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in PRONOUN_LEAKS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 5. Block Hallucinated Inventions
        for pat in HALLUCINATED_INVENTIONS:
            if re.search(pat, text, flags=re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    sanitized_text=fallback_statement,
                    reason=f"Hallucinated backstory detected: {pat}"
                )

        # 6. Block Repetitive Teacher Prompts
        for pat in REPETITIVE_DOCTOR_PROMPTS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

        # 7. Contradiction Validation Against Active Session State
        if session_state:
            gender = str(session_state.demographics.get("gender") or "").strip().lower()
            is_male = gender in ["male", "m", "man"]
            lower_text = text.lower()

            # A. Gender Contradiction (Male patient claiming female conditions)
            if is_male and any(k in lower_text for k in ["i have pcod", "i have pcos", "my period", "my menstrual", "i am pregnant", "my ovaries"]):
                return ValidationResult(
                    is_valid=False,
                    sanitized_text="I'm a male patient, doctor, so that doesn't apply to me.",
                    reason="Contradiction: male patient claiming female condition"
                )

            # B. Contradiction of Primary Chest Pain
            if session_state.character and any(k in session_state.character.lower() for k in ["elephant", "pressure", "crushing", "squeezing", "pain"]):
                if any(denial in lower_text for denial in ["i don't have chest pain", "i don't have any pain", "no chest pain", "my chest feels completely fine"]):
                    return ValidationResult(
                        is_valid=False,
                        sanitized_text=fallback_statement,
                        reason="Contradiction: patient denied active chest pain"
                    )

            # C. Contradiction of Negative Facts
            if fallback_statement.startswith("No,") or fallback_statement.startswith("I don't have"):
                if any(k in lower_text for k in ["yes, i have", "yes, i do", "i definitely have"]):
                    return ValidationResult(
                        is_valid=False,
                        sanitized_text=fallback_statement,
                        reason="Contradiction: affirmed a negative symptom"
                    )

        # 8. Truncate overly verbose responses to 2-3 sentences max
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 3:
            text = " ".join(sentences[:2])

        if not text:
            return ValidationResult(is_valid=False, sanitized_text=fallback_statement, reason="Sanitization emptied response")

        return ValidationResult(is_valid=True, sanitized_text=text)
