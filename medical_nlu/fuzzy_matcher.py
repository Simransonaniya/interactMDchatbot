"""
InteractMD — Controlled Medical Medication Fuzzy Matcher.
Performs confidence-scored approximate matching of medication terms against the
medication vocabulary while preventing non-medical word hallucinations.
"""

from typing import Optional, Dict, Any, List, Tuple
from medical_nlu.medication_catalog import MedicationCatalog


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Computes similarity ratio in range [0.0, 1.0]."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


class MedicationFuzzyMatcher:
    """
    Controlled fuzzy matching engine with adaptive thresholds:
    - Length < 4: Exact match only (prevent 'in', 'on', 'at' matching drug names)
    - Length 4-6: Max 1 edit distance, similarity >= 0.80
    - Length 7+: Max 2 edit distances, similarity >= 0.78
    """

    def __init__(self):
        self.catalog = MedicationCatalog.get_instance()

    def match(self, token: str, min_confidence: float = 0.78) -> Optional[Dict[str, Any]]:
        norm = token.lower().strip()
        if len(norm) < 4:
            # Short tokens must be exact
            canonical = self.catalog.alias_to_canonical.get(norm)
            if canonical:
                return {
                    "matched_token": token,
                    "canonical": canonical,
                    "confidence": 1.0,
                    "method": "exact_short"
                }
            return None

        # 1. Exact alias match
        if norm in self.catalog.alias_to_canonical:
            return {
                "matched_token": token,
                "canonical": self.catalog.alias_to_canonical[norm],
                "confidence": 1.0,
                "method": "exact_alias"
            }

        # 2. Fuzzy scan over registered terms
        best_match = None
        best_sim = 0.0
        best_term = ""

        # Pre-filter by length difference <= 2
        norm_len = len(norm)
        for term in self.catalog.exact_terms:
            term_len = len(term)
            if abs(term_len - norm_len) > 2:
                continue

            sim = similarity_ratio(norm, term)
            if sim > best_sim:
                best_sim = sim
                best_term = term

        # Evaluate threshold based on token length
        max_allowed_dist = 1 if norm_len <= 6 else 2
        if best_term and best_sim >= min_confidence:
            dist = levenshtein_distance(norm, best_term)
            if dist <= max_allowed_dist:
                canonical = self.catalog.alias_to_canonical.get(best_term, best_term)
                return {
                    "matched_token": token,
                    "canonical": canonical,
                    "confidence": round(best_sim, 2),
                    "method": "controlled_fuzzy_edit_distance",
                    "target_term": best_term
                }

        return None
