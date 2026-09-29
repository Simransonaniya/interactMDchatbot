"""
InteractMD — Advanced Medical Entity Extraction Layer.
Detects clinical and contextual entities from normalized and raw clinician input:
- MEDICATION (canonical drug names, classes, generic formulations: pill, tablet, medicine, meds)
- ACTIVE_INGREDIENT (individual active drug substances, preserving combinations)
- BRAND_NAME (commercial drug product names)
- DOSAGE (numeric dose values, e.g. 5, 20, 650)
- UNIT (dose units: mg, mcg, g, ml, puffs, tablets, drops, units)
- FREQUENCY (daily, once a day, twice a day, regularly, morning, night, as needed, prn)
- ROUTE (oral, sublingual, inhaler/inhalation, intravenous, injection, topical)
- DURATION (45 minutes, 2 hours, 6 years, 25 years, this week, last week)
- FORM (tablet, pill, capsule, inhaler, injection, patch, spray, drops)
- REASON (hypertension, blood pressure, cholesterol, pain, asthma)
- ADHERENCE (missed, skipped, regular, forgot, compliant, stopped)
- TEMPORAL (today, yesterday, this morning, tonight, last night, this week, last week, currently, normally, usually)
- NEGATION (not taking, don't take, stopped, never took, no medications)
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from medical_nlu.medication_catalog import MedicationCatalog, MedicationConcept
from medical_nlu.fuzzy_matcher import MedicationFuzzyMatcher


@dataclass
class MedicalEntity:
    text: str
    normalized: str
    type: str  # MEDICATION, ACTIVE_INGREDIENT, BRAND_NAME, DOSAGE, UNIT, FREQUENCY, ROUTE, DURATION, FORM, REASON, ADHERENCE, TEMPORAL, NEGATION, FOOD_MEAL, SYMPTOM, DISEASE, BODY_PART, LAB_TEST, PROCEDURE
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


# Clinical entity regex patterns
DOSAGE_PATTERN = r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|puffs?|drops?|tablets?|capsules?|pills?|units?)\b"
STANDALONE_NUMBER_DOSE = r"\b(?:take|dose of|strength of|about)\s+(\d+(?:\.\d+)?)\b"
FREQUENCY_PATTERN = r"\b(daily|once\s+(?:a\s+)?day|twice\s+(?:a\s+)?day|twice\s+daily|three\s+times\s+(?:a\s+)?day|four\s+times\s+(?:a\s+)?day|every\s+(?:morning|day|night|\d+\s*hours?)|as\s+needed|prn|bid|tid|qid|regularly|nightly|at\s+night|in\s+the\s+morning|every\s+morning|every\s+night)\b"
ROUTE_PATTERN = r"\b(oral|orally|by\s+mouth|sublingual|sublingually|under\s+the\s+tongue|inhaler|inhalation|inhaled|intravenous|iv|injection|injected|subcutaneous|sc|topical|patch|drops?)\b"
FORM_PATTERN = r"\b(tablets?|pills?|capsules?|inhalers?|puffer|injections?|syrup|suspension|spray|cream|ointment|gel|patch|drops?)\b"
DURATION_PATTERN = r"\b(\d+\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?)|about\s+\d+\s*(?:minutes?|hours?)|for\s+\d+\s*(?:years?|months?|days?|weeks?)|since\s+\d+\s*(?:years?|months?))\b"
TEMPORAL_PATTERN = r"\b(today|yesterday|this\s+morning|tonight|last\s+night|this\s+week|last\s+week|recently|before\s+coming\s+here|currently|normally|usually|at\s+present|right\s+now)\b"
NEGATION_PATTERN = r"\b(do\s+not\s+take|don't\s+take|dont\s+take|stopped\s+taking|stopped|not\s+taking|never\s+took|never\s+taken|no\s+medications?|no\s+medicines?|do\s+not\s+use|don't\s+use|not\s+on\s+any)\b"
ADHERENCE_PATTERN = r"\b(miss(ed)?|skip(ped)?|forget|forgotten|forgetting|regularly|every\s+day|frequently\s+miss|skip\s+doses|adherent|compliance|non-compliant|stopped)\b"
REASON_PATTERN = r"\b(for\s+(?:my\s+|the\s+|your\s+)?(blood\s+pressure|hypertension|cholesterol|chest\s+pain|angina|heart|pain|fever|headache|anxiety|depression|asthma|breathing))\b"

FOOD_MEAL_PATTERNS = [
    r"\b(breakfast|lunch|dinner|supper|brunch|snack|snacks|meal|meals|food)\b"
]

SYMPTOM_PATTERNS = [
    (r"\b(chest\s+pain|pressure|crushing\s+pressure|tightness|heaviness|vice|elephant)\b", "chest_pain"),
    (r"\b(short(?:ness)?\s+of\s+breath|breathless|dyspnea|hard\s+to\s+breathe|trouble\s+breathing|winded)\b", "shortness_of_breath"),
    (r"\b(cold\s+sweats?|sweat(?:ing)?|clammy|diaphoresis|perspir[a-z]*)\b", "sweating"),
    (r"\b(dizzy|dizziness|lightheaded|light-headed|faint|syncope)\b", "dizziness"),
    (r"\b(nausea|nauseous|queasy|sick\s+to\s+(?:my|your)\s+stomach)\b", "nausea"),
    (r"\b(vomit(?:ing)?|threw\s+up|throw\s+up|puke|puking)\b", "vomiting"),
    (r"\b(palpitations?|heart\s+racing|fluttering|fast\s+heart\s*beat)\b", "palpitations"),
    (r"\b(fever|feverish|chills|shivering)\b", "fever"),
    (r"\b(cough(?:ing)?|sputum|phlegm)\b", "cough"),
    (r"\b(wheez(?:ing)?|whistling\s+sound)\b", "wheezing"),
    (r"\b(body\s+(?:pain|aches?)|muscle\s+(?:pain|aches?)|myalgia)\b", "body_aches"),
    (r"\b(headache|head\s+pain|migraine)\b", "headache"),
    (r"\b(back\s+pain|backache)\b", "back_pain"),
    (r"\b(swelling|edema|swollen\s+(?:legs?|feet|ankles?))\b", "swelling"),
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
    (r"\b(back|between\s+(?:the\s+)?shoulder\s+blades)\b", "back"),
    (r"\b(stomach|abdomen|belly)\b", "abdomen"),
    (r"\b(legs?|feet|ankles?)\b", "legs"),
]

LAB_TEST_PATTERNS = [
    (r"\b(12-lead\s+)?(ecg|ekg)\b", "ecg"),
    (r"\b(troponin(?:\s*i)?|cardiac\s+enzymes?)\b", "troponin"),
    (r"\b(chest\s+x-ray|cxr|x-ray)\b", "chest_xray"),
    (r"\b(blood\s+test|bloodwork|labs|cbc|bmp)\b", "blood_tests"),
    (r"\b(ct\s+scan|ultrasound)\b", "imaging"),
]

PROCEDURE_PATTERNS = [
    (r"\b(vital\s+signs|vitals|blood\s+pressure|heart\s+rate|pulse|temp|temperature)\b", "vital_signs"),
    (r"\b(physical\s+exam(?:ination)?|cardiovascular\s+exam|chest\s+exam|listen\s+to\s+(?:your\s+)?(?:heart|lungs|chest)|auscultat[a-z]*)\b", "physical_exam"),
]

LIFESTYLE_BEHAVIOR_PATTERNS = [
    (r"\b(caffeine|coffee|tea|energy\s+drinks?)\b", "caffeine"),
    (r"\b(regular\s+sleep|sleep|restful\s+sleep|bedtime|hours\s+of\s+sleep)\b", "sleep"),
    (r"\b(lightweight|light\s+exercise|exercise|workout|gym|walking|jogging|physical\s+activity)\b", "exercise"),
    (r"\b(regular\s+meals?|regular\s+means?|healthy\s+diet|dietary\s+changes?|reduce\s+salt)\b", "diet"),
    (r"\b(stress|reduce\s+stress|stress\s+management|meditation|relaxation|relaxation\s+exercise)\b", "stress_reduction"),
    (r"\b(smoke|smoking|cigarettes?|tobacco|vape|vaping)\b", "smoking"),
    (r"\b(alcohol|drinking|wine|beer|liquor)\b", "alcohol"),
]


class MedicalEntityExtractor:
    """Production-grade medical entity extractor with multi-attribute parsing."""

    def __init__(self):
        self.catalog = MedicationCatalog.get_instance()
        self.fuzzy_matcher = MedicationFuzzyMatcher()

    def extract_entities(self, normalized_text: str, raw_text: Optional[str] = None) -> List[MedicalEntity]:
        entities: List[MedicalEntity] = []
        text_to_scan = normalized_text.lower()
        raw = raw_text or normalized_text

        # 1. MEDICATION & ACTIVE INGREDIENT & BRAND extraction via catalog
        sorted_terms = sorted(self.catalog.exact_terms, key=len, reverse=True)
        matched_spans = []

        for term in sorted_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            for m in re.finditer(pattern, text_to_scan):
                span = (m.start(), m.end())
                # Avoid overlapping spans
                if any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                    continue
                matched_spans.append(span)

                canonical = self.catalog.alias_to_canonical.get(term, term)
                concept = self.catalog.concepts.get(canonical)

                meta = {
                    "drug_class": concept.drug_class if concept else "general",
                    "active_ingredients": concept.active_ingredients if concept else [canonical],
                    "brand_names": concept.brand_names if concept else [],
                    "dosage_forms": concept.dosage_forms if concept else ["tablet"],
                    "is_combination": concept.is_combination if concept else False,
                    "indications_purpose": concept.indications_purpose if concept else "",
                    "common_side_effects": concept.common_side_effects if concept else []
                }

                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=canonical,
                    type="MEDICATION",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.98,
                    metadata=meta
                ))

                # Also emit ACTIVE_INGREDIENT entities
                if concept and concept.active_ingredients:
                    for ing in concept.active_ingredients:
                        entities.append(MedicalEntity(
                            text=ing,
                            normalized=ing,
                            type="ACTIVE_INGREDIENT",
                            start=m.start(),
                            end=m.end(),
                            confidence=0.98,
                            metadata={"parent_medication": canonical}
                        ))

                # Emit BRAND_NAME entity if matched term was a brand name
                if concept and any(b.lower() == term for b in concept.brand_names):
                    entities.append(MedicalEntity(
                        text=m.group(0),
                        normalized=term.title(),
                        type="BRAND_NAME",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.98,
                        metadata={"canonical": canonical}
                    ))

        # If no exact catalog match, attempt controlled fuzzy matching on words
        if not any(e.type == "MEDICATION" for e in entities):
            words = text_to_scan.split()
            for w in words:
                clean_w = re.sub(r"[^\w]", "", w)
                if len(clean_w) >= 4:
                    f_match = self.fuzzy_matcher.match(clean_w)
                    if f_match and f_match["confidence"] >= 0.80:
                        canonical = f_match["canonical"]
                        concept = self.catalog.concepts.get(canonical)
                        start_idx = text_to_scan.find(clean_w)
                        end_idx = start_idx + len(clean_w) if start_idx != -1 else 0

                        entities.append(MedicalEntity(
                            text=clean_w,
                            normalized=canonical,
                            type="MEDICATION",
                            start=start_idx,
                            end=end_idx,
                            confidence=f_match["confidence"],
                            metadata={
                                "drug_class": concept.drug_class if concept else "general",
                                "active_ingredients": concept.active_ingredients if concept else [canonical],
                                "fuzzy_matched": True,
                                "method": f_match.get("method")
                            }
                        ))
                        break

        # Fallback generic medication token extraction
        if not any(e.type == "MEDICATION" for e in entities):
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

        # 2. DOSAGE & UNIT extraction
        for m in re.finditer(DOSAGE_PATTERN, text_to_scan):
            num_val = m.group(1)
            unit_val = m.group(2)
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=f"{num_val} {unit_val}",
                type="DOSAGE",
                start=m.start(),
                end=m.end(),
                confidence=0.95,
                metadata={"dose": num_val, "unit": unit_val}
            ))
            entities.append(MedicalEntity(
                text=unit_val,
                normalized=unit_val,
                type="UNIT",
                start=m.start(2),
                end=m.end(2),
                confidence=0.95
            ))

        # 3. FREQUENCY extraction
        for m in re.finditer(FREQUENCY_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="FREQUENCY",
                start=m.start(),
                end=m.end(),
                confidence=0.92
            ))

        # 4. ROUTE extraction
        for m in re.finditer(ROUTE_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="ROUTE",
                start=m.start(),
                end=m.end(),
                confidence=0.92
            ))

        # 5. FORM extraction
        for m in re.finditer(FORM_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="FORM",
                start=m.start(),
                end=m.end(),
                confidence=0.92
            ))

        # 6. DURATION extraction
        for m in re.finditer(DURATION_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="DURATION",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 7. TEMPORAL extraction
        for m in re.finditer(TEMPORAL_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="TEMPORAL",
                start=m.start(),
                end=m.end(),
                confidence=0.95
            ))

        # 8. NEGATION extraction
        for m in re.finditer(NEGATION_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="NEGATION",
                start=m.start(),
                end=m.end(),
                confidence=0.95
            ))

        # 9. ADHERENCE extraction
        for m in re.finditer(ADHERENCE_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(0),
                type="ADHERENCE",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 10. REASON extraction
        for m in re.finditer(REASON_PATTERN, text_to_scan):
            entities.append(MedicalEntity(
                text=m.group(0),
                normalized=m.group(2) if m.group(2) else m.group(0),
                type="REASON",
                start=m.start(),
                end=m.end(),
                confidence=0.90
            ))

        # 11. FOOD_MEAL extraction
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

        # 12. SYMPTOM extraction
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

        # 13. DISEASE extraction
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

        # 14. BODY_PART extraction
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

        # 15. LAB_TEST extraction
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

        # 16. PROCEDURE extraction
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

        # 17. LIFESTYLE_BEHAVIOR extraction
        for pat, norm in LIFESTYLE_BEHAVIOR_PATTERNS:
            for m in re.finditer(pat, text_to_scan):
                entities.append(MedicalEntity(
                    text=m.group(0),
                    normalized=norm,
                    type="LIFESTYLE_BEHAVIOR",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95
                ))

        return entities
