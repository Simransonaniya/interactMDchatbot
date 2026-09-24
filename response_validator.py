"""
InteractMD — Patient Response Validator & Sanitizer.
Ensures first-person patient voice, blocks hidden diagnostic leaks, enforces length constraints,
prevents factual hallucination / mental health causal fabrication, and shields system prompts.
"""

import re
from typing import Dict, Any, List, Tuple, Optional


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
    def validate(raw_response: Optional[str], fallback_statement: str) -> ValidationResult:
        if not raw_response or not raw_response.strip():
            return ValidationResult(is_valid=False, sanitized_text=fallback_statement, reason="Empty response")

        text = raw_response.strip()

        # 1. Clean markdown / JSON artifacts
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.replace('"', '').replace("'", "'").strip()

        # 2. Check for hidden diagnosis leaks
        t_low = text.lower()
        for diag in HIDDEN_DIAGNOSIS_TERMS:
            if diag in t_low:
                return ValidationResult(
                    is_valid=False,
                    sanitized_text=fallback_statement,
                    reason=f"Hidden medical diagnostic leakage detected: {diag}"
                )

        # 3. Check for system prompt / JSON leakage
        if any(leak in t_low for leak in ["system prompt", "case json", "ground truth", "evaluator", "database", "clinical fact:"]):
            return ValidationResult(
                is_valid=False,
                sanitized_text=fallback_statement,
                reason="System prompt / database leakage detected"
            )

        # 4. Check for hallucinated causal / lifestyle inventions
        for hall_pat in HALLUCINATED_INVENTIONS:
            if re.search(hall_pat, t_low):
                return ValidationResult(
                    is_valid=False,
                    sanitized_text=fallback_statement,
                    reason=f"Hallucinated patient invention detected: {hall_pat}"
                )

        # 5. Sanitize 3rd person to 1st person
        for pattern, replacement in THIRD_PERSON_VIOLATIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in PRONOUN_LEAKS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 6. Remove repetitive doctor-guiding questions
        for pattern in REPETITIVE_DOCTOR_PROMPTS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # 7. Length enforcement (1-3 sentences max)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) > 3:
            text = " ".join(sentences[:2])
            if not text.endswith(('.', '!', '?')):
                text += "."

        # 8. Ensure proper capitalization
        if text:
            text = text[0].upper() + text[1:]

        # 9. Check minimal validity
        if len(text.split()) < 2:
            return ValidationResult(is_valid=False, sanitized_text=fallback_statement, reason="Response too short")

        return ValidationResult(is_valid=True, sanitized_text=text)

    @staticmethod
    def validate_and_sanitize(
        raw_response: Optional[str],
        expected_facet: str = "",
        fallback_statement: str = ""
    ) -> str:
        res = PatientResponseValidator.validate(raw_response, fallback_statement)
        return res.sanitized_text
