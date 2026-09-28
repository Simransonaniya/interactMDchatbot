"""
InteractMD — Medical Entity Extraction Layer.
Detects clinical and contextual entities from normalized clinician input:
- MEDICATION (canonical drug names, classes, generic formulations: pill, tablet, medicine, meds)
- FOOD_MEAL (breakfast, lunch, dinner, supper, food, meal, snack)
- DOSAGE (5mg, 20 mg, 1 tablet, 2 puffs)
- FREQUENCY (daily, once a day, twice a day, regularly, morning, night)
- DURATION (45 minutes, 2 hours, 6 years, 25 years, today, yesterday)
- SYMPTOM (chest pain, shortness of breath, sweating, dizziness, nausea, vomiting, etc.)
- DISEASE (hypertension, heart attack, asthma, pcod, diabetes, etc.)
- BODY_PART (chest, jaw, arm, back, shoulder, neck, stomach, legs)
- LAB_TEST (ecg, ekg, troponin, chest x-ray, blood test, labs)
- PROCEDURE (vitals, physical examination, auscultation)
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from medical_nlu.medication_lexicon import MedicationLexicon


@dataclass
class MedicalEntity:
    text: str
    normalized: str
    type: str  # MEDICATION, FOOD_MEAL, DOSAGE, FREQUENCY, DURATION, SYMPTOM, DISEASE, BODY_PART, LAB_TEST, PROCEDURE
    start: int
    end: int
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


# Regex patterns for clinical entities
DOSAGE_PATTERN = r"\b(\d+(\.\d+)?\s*(mg|mcg|g|ml|puffs?|drops?|tablets?|capsules?|pills?)|one\s+(tablet|pill|puff)|two\s+(tablets|pills|puffs))\b"
FREQUENCY_PATTERN = r"\b(daily|once\s+(a\s+)?day|twice\s+(a\s+)?day|three\s+times\s+(a\s+)?day|every\s+(morning|day|night|\d+\s*hours?)|as\s+needed|prn|bid|tid|qid|regularly|morning|night)\b"
DURATION_PATTERN = r"\b(\d+\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?)|about\s+\d+\s*(minutes?|hours?)|yesterday|today|this\s+morning|last\s+night|since\s+morning)\b"

FOOD_MEAL_PATTERNS = [
    r"\b(breakfast|lunch|dinner|supper|brunch|snack|snacks|meal|meals|food)\b"
]

SYMPTOM_PATTERNS = [
    (r"\b(chest\s+pain|pressure|crushing\s+pressure|tightness|heaviness|vice|elephant)\b", "chest_pain"),
    (r"\b(short(ness)?\s+of\s+breath|breathless|dyspnea|hard\s+to\s+breathe|trouble\s+breathing|winded)\b", "shortness_of_breath"),
    (r"\b(cold\s+sweats?|sweat(ing)?|clammy|diaphoresis|perspir[a-z]*)\b", "sweating"),
    (r"\b(dizzy|dizziness|lightheaded|light-headed|faint|syncope)\b", "dizziness"),
    (r"\b(nausea|nauseous|queasy|sick\s+to\s+(my|your)\s+stomach)\b", "nausea"),
    (r"\b(vomit(ing)?|threw\s+up|throw\s+up|puke|puking)\b", "vomiting"),
    (r"\b(palpitations?|heart\s+racing|fluttering|fast\s+heart\s*beat)\b", "palpitations"),
    (r"\b(fever|feverish|chills|shivering)\b", "fever"),
    (r"\b(cough(ing)?|sputum|phlegm)\b", "cough"),
    (r"\b(wheez(ing)?|whistling\s+sound)\b", "wheezing"),
    (r"\b(body\s+(pain|aches?)|muscle\s+(pain|aches?)|myalgia)\b", "body_aches"),
    (r"\b(headache|head\s+pain|migraine)\b", "headache"),
    (r"\b(back\s+pain|backache)\b", "back_pain"),
    (r"\b(swelling|edema|swollen\s+(legs?|feet|ankles?))\b", "swelling"),
]

DISEASE_PATTERNS = [
    (r"\b(heart\s+attack|myocardial\s+infarction|mi|stemi|angina|cardiac)\b", "cardiac_disease"),
    (r"\b(hypertension|high\s+blood\s+pressure|htn)\b", "hypertension"),
    (r"\b(hyperlipidemia|high\s+cholesterol|cholesterol)\b", "hyperlipidemia"),
    (r"\b(asthma|copd|bronchitis)\b", "respiratory_disease"),
    (r"\b(diabetes|type\s+2\s+diabetes|diabetic)\b", "diabetes"),
    (r"\b(pcod|pcos|polycystic\s+ovary)\b", "pcod"),
    (r"\b(cancer|tumor|malignancy)\b", "cancer"),
    (r"\b(vitiligo|depigmentation)\b", "vitiligo"),
]

BODY_PART_PATTERNS = [
    (r"\b(chest|substernal|retrosternal)\b", "chest"),
    (r"\b(left\s+arm|right\s+arm|arm|arms)\b", "arm"),
    (r"\b(jaw|lower\s+jaw)\b", "jaw"),
    (r"\b(left\s+shoulder|shoulder|shoulders)\b", "shoulder"),
    (r"\b(neck)\b", "neck"),
    (r"\b(back|between\s+(the\s+)?shoulder\s+blades)\b", "back"),
    (r"\b(stomach|abdomen|belly)\b", "abdomen"),
    (r"\b(legs?|feet|ankles?)\b", "legs"),
]

LAB_TEST_PATTERNS = [
    (r"\b(12-lead\s+)?(ecg|ekg)\b", "ecg"),
    (r"\b(troponin(\s*i)?|cardiac\s+enzymes?)\b", "troponin"),
    (r"\b(chest\s+x-ray|cxr|x-ray)\b", "chest_xray"),
    (r"\b(blood\s+test|bloodwork|labs|cbc|bmp)\b", "blood_tests"),
    (r"\b(ct\s+scan|ultrasound)\b", "imaging"),
]

PROCEDURE_PATTERNS = [
    (r"\b(vital\s+signs|vitals|blood\s+pressure|heart\s+rate|pulse|temp|temperature)\b", "vital_signs"),
    (r"\b(physical\s+exam(ination)?|cardiovascular\s+exam|chest\s+exam|listen\s+to\s+(your\s+)?(heart|lungs|chest)|auscultat[a-z]*)\b", "physical_exam"),
]


class MedicalEntityExtractor:
    """Fast, accurate entity extraction using the medication lexicon and clinical rules."""

    def __init__(self):
        self.lexicon = MedicationLexicon.get_instance()

    def extract_entities(self, normalized_text: str, raw_text: Optional[str] = None) -> List[MedicalEntity]:
        entities: List[MedicalEntity] = []
        text_to_scan = normalized_text.lower()
        raw = raw_text or normalized_text

        # 1. MEDICATION extraction via lexicon
        med_match = self.lexicon.match_medication(text_to_scan)
        if med_match:
            entities.append(MedicalEntity(
                text=med_match["matched_text"],
                normalized=med_match["canonical"],
                type="MEDICATION",
                start=med_match["start"],
                end=med_match["end"],
                confidence=0.98,
                metadata={"drug_class": med_match["drug_class"]}
            ))
        else:
            # Fallback for generic medication phrases
            gen_med_pat = r"\b(medicine|medicines|medication|medications|meds|tablet|tablets|pill|pills|capsule|capsules|drugs?|prescriptions?|daily\s+meds)\b"
            m = re.search(gen_med_pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized="medicine",
                    type="MEDICATION",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95,
                    metadata={"drug_class": "generic_term"}
                ))

        # 2. FOOD_MEAL extraction
        for pat in FOOD_MEAL_PATTERNS:
            for m in re.finditer(pat, text_to_scan):
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=m.group(0),
                    type="FOOD_MEAL",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        # 3. DOSAGE extraction
        for m in re.finditer(DOSAGE_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="DOSAGE",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 4. FREQUENCY extraction
        for m in re.finditer(FREQUENCY_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="FREQUENCY",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 5. DURATION extraction
        for m in re.finditer(DURATION_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="DURATION",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 6. SYMPTOM extraction
        for pat, norm in SYMPTOM_PATTERNS:
            m = re.search(pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="SYMPTOM",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        # 7. DISEASE extraction
        for pat, norm in DISEASE_PATTERNS:
            m = re.search(pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="DISEASE",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        # 8. BODY_PART extraction
        for pat, norm in BODY_PART_PATTERNS:
            m = re.search(pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="BODY_PART",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        # 9. LAB_TEST extraction
        for pat, norm in LAB_TEST_PATTERNS:
            m = re.search(pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="LAB_TEST",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        # 10. PROCEDURE extraction
        for pat, norm in PROCEDURE_PATTERNS:
            m = re.search(pat, text_to_scan)
            if m:
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="PROCEDURE",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        return entities
