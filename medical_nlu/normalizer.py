"""
InteractMD — Medical Text Normalization Layer.
Standardizes clinical and colloquial learner text without altering the visible UI transcript.
Handles:
- Case normalization & whitespace stripping
- Medical contractions expansion
- Common clinical slang & spelling variants (e.g. medcian -> medicine, paracetomol -> paracetamol)
- Clinical grammar corrections (e.g. "had you breakfast" -> "did you have breakfast")
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class NormalizedText:
    raw_text: str
    normalized_text: str
    tokens: List[str]
    corrections: List[Dict[str, str]] = field(default_factory=list)


CONTRACTIONS = {
    r"\bdon't\b": "do not",
    r"\bdont\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bcant\b": "cannot",
    r"\bi'm\b": "i am",
    r"\bim\b": "i am",
    r"\bi've\b": "i have",
    r"\bive\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\blet's\b": "let us",
    r"\blets\b": "let us",
    r"\bwhat's\b": "what is",
    r"\bwhats\b": "what is",
    r"\bthat's\b": "that is",
    r"\bthats\b": "that is",
    r"\bwhere's\b": "where is",
    r"\bhow's\b": "how is",
    r"\bit's\b": "it is",
    r"\bits\b": "it is",
    r"\byou're\b": "you are",
    r"\byoure\b": "you are",
    r"\byou've\b": "you have",
    r"\byouve\b": "you have",
    r"\bdid't\b": "did not",
    r"\bdidn't\b": "did not",
    r"\bdidnt\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhasnt\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhavent\b": "have not",
    r"\bwon't\b": "will not",
    r"\bwont\b": "will not",
    r"\bwe'll\b": "we will",
    r"\bwe're\b": "we are",
    r"\bwe'd\b": "we would",
    r"\bthey're\b": "they are",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
}

# Spelling corrections & shorthand normalization
TYPO_MAPPINGS: List[Tuple[str, str]] = [
    # Common medication typos & variations
    (r"\bmedc[a-z]*\b", "medicine"),        # medcian, medcines, medcine, medcins, medcians
    (r"\bmedic[a-df-z][a-z]*\b", "medicine"), # medician, medican, medicin, medicne, medicen
    (r"\bmedec[a-z]*\b", "medicine"),       # medecine, medecin, medecines
    (r"\bmedicat[a-z]*\b", "medication"),   # medication, medications, medicationn, medicatns
    (r"\bmeds?\b", "medication"),           # med, meds
    (r"\btabs?\b", "tablet"),               # tab, tabs
    (r"\bpills?\b", "pill"),                # pill, pills
    (r"\bparacetomol\b", "paracetamol"),
    (r"\bparacetmol\b", "paracetamol"),
    (r"\bparacitamol\b", "paracetamol"),
    (r"\bparacetemol\b", "paracetamol"),
    (r"\bniciplus\b", "nicip plus"),
    (r"\bnicip-plus\b", "nicip plus"),
    (r"\bnishchit\s+plus\b", "nicip plus"),
    (r"\bnishchit\b", "nicip plus"),
    (r"\bsertrakine\b", "sertraline"),
    (r"\bescita;pram\b", "escitalopram"),
    (r"\bescitapram\b", "escitalopram"),
    (r"\bamlodipin\b", "amlodipine"),
    (r"\bamlodepine\b", "amlodipine"),
    (r"\batorvastin\b", "atorvastatin"),
    (r"\batorvastatine\b", "atorvastatin"),
    (r"\batrovastatin\b", "atorvastatin"),
    (r"\basprin\b", "aspirin"),
    (r"\bclopidogel\b", "clopidogrel"),
    (r"\bpan\s*40\b", "pan 40"),
    (r"\bdolo\s*650\b", "dolo 650"),
    
    # Common symptom & clinical typos
    (r"\bnausious\b", "nauseous"),
    (r"\bdypsnea\b", "dyspnea"),
    (r"\bdispnea\b", "dyspnea"),
    (r"\bbrethless\b", "breathless"),
    (r"\bdizzyness\b", "dizziness"),
    (r"\bpalpatations?\b", "palpitations"),
    (r"\bpresure\b", "pressure"),
    (r"\btitgness\b", "tightness"),
    (r"\btightnes\b", "tightness"),
    (r"\bseveir\b", "severe"),
    (r"\bradit[a-z]*\b", "radiate"),
    
    # Common colloquial / non-native grammar phrasing
    (r"\bregular\s+means\b", "regular meals"),
    (r"\ba\s+anxiety\b", "anxiety"),
    (r"\bhad\s+you\s+breakfast\b", "did you have breakfast"),
    (r"\bhad\s+you\s+lunch\b", "did you have lunch"),
    (r"\bhad\s+you\s+dinner\b", "did you have dinner"),
    (r"\bwhat\s+was\s+you\s+eat\b", "what did you eat"),
    (r"\bwhat\s+you\s+eat\b", "what did you eat"),
    (r"\bwhat\s+you\s+had\b", "what did you have"),
    (r"\bcan\s+you\s+tell\s+me\s+what\s+medication\s+you\s+take\b", "what medications do you take"),
    (r"\bwhat\s+medicine\s+did\s+you\s+eat\b", "what medicine did you take"),
    (r"\bdid\s+you\s+ate\b", "did you eat"),
    (r"\bdid\s+you\s+taken\b", "did you take"),
    (r"\bhave\s+you\s+take\b", "have you taken"),
]


class MedicalNormalizer:
    """Fast, deterministic, safe clinical text normalization."""

    @classmethod
    def normalize(cls, raw_query: str) -> NormalizedText:
        if not raw_query:
            return NormalizedText(raw_text="", normalized_text="", tokens=[], corrections=[])

        text = raw_query.strip()
        corrections = []

        # 1. Lowercase
        normalized = text.lower()

        # 2. Expand contractions (e.g. don't -> do not, what's -> what is)
        for pattern, replacement in CONTRACTIONS.items():
            if re.search(pattern, normalized):
                corrections.append({"type": "contraction", "pattern": pattern, "replacement": replacement})
                normalized = re.sub(pattern, replacement, normalized)

        # 3. Standardize punctuation / remove noisy symbols
        normalized = re.sub(r"[^\w\s\?\-\/\.\,]", " ", normalized)

        # 4. Apply medical typo corrections and clinical phrasing mappings
        for pattern, replacement in TYPO_MAPPINGS:
            if re.search(pattern, normalized):
                corrections.append({"type": "typo_fix", "pattern": pattern, "replacement": replacement})
                normalized = re.sub(pattern, replacement, normalized)

        # 5. Clean excess whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        tokens = normalized.split()

        return NormalizedText(
            raw_text=raw_query,
            normalized_text=normalized,
            tokens=tokens,
            corrections=corrections
        )
