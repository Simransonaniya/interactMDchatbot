"""
InteractMD — Medication Lexicon & Terminology Repository.
Provides fast, indexed lookup for:
- Canonical medication names
- Aliases, brand names, and drug classes
- Common misspellings and shorthand
- Extensible JSON-backed configuration
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass


@dataclass
class MedicationEntry:
    canonical: str
    drug_class: str
    aliases: List[str]
    misspellings: List[str]


class MedicationLexicon:
    _instance: Optional["MedicationLexicon"] = None

    def __init__(self, json_path: Optional[str] = None):
        self.entries: Dict[str, MedicationEntry] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.exact_terms: Set[str] = set()
        
        if json_path is None:
            # Default to chatbot/data/lexicons/medications.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(base_dir, "data", "lexicons", "medications.json")
        
        self.load(json_path)

    @classmethod
    def get_instance(cls) -> "MedicationLexicon":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"[MedicationLexicon Warning] Lexicon file not found at {file_path}, initializing defaults.")
            self._load_fallback_defaults()
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for key, val in data.items():
                canonical = val.get("canonical", key)
                drug_class = val.get("class", "general")
                aliases = val.get("aliases", [canonical])
                misspellings = val.get("common_misspellings", [])
                
                entry = MedicationEntry(
                    canonical=canonical,
                    drug_class=drug_class,
                    aliases=aliases,
                    misspellings=misspellings
                )
                self.entries[canonical] = entry

                # Index all alias and misspelling terms
                for alias in aliases:
                    norm_alias = alias.lower().strip()
                    self.alias_to_canonical[norm_alias] = canonical
                    self.exact_terms.add(norm_alias)

                for misspelling in misspellings:
                    norm_miss = misspelling.lower().strip()
                    self.alias_to_canonical[norm_miss] = canonical
                    self.exact_terms.add(norm_miss)

        except Exception as e:
            print(f"[MedicationLexicon Error] Failed to load lexicon: {e}")
            self._load_fallback_defaults()

    def _load_fallback_defaults(self):
        defaults = [
            ("paracetamol", "analgesic", ["paracetamol", "paracetomol", "acetaminophen", "tylenol", "crocin", "dolo", "combiflam"]),
            ("amlodipine", "ccb", ["amlodipine", "amlodipin", "norvasc", "amlodipine 5mg", "amlodipine 5 mg"]),
            ("atorvastatin", "statin", ["atorvastatin", "atorva", "lipitor", "atorvastatin 20mg", "atorvastatin 20 mg", "statin"]),
            ("aspirin", "antiplatelet", ["aspirin", "asprin", "ecosprin", "disprin"]),
            ("nicip plus", "nsaid", ["nicip plus", "niciplus", "nicip", "nishchit plus", "nishchit"]),
            ("sertraline", "ssri", ["sertraline", "sertrakine", "zoloft"]),
            ("escitalopram", "ssri", ["escitalopram", "escita;pram", "lexapro"]),
            ("inhaler", "bronchodilator", ["inhaler", "blue inhaler", "salbutamol", "albuterol"]),
            ("medication", "generic", ["medicine", "medicines", "medication", "medications", "meds", "tablet", "tablets", "pill", "pills", "capsule", "drug", "drugs"])
        ]
        for canonical, drug_class, aliases in defaults:
            entry = MedicationEntry(canonical=canonical, drug_class=drug_class, aliases=aliases, misspellings=[])
            self.entries[canonical] = entry
            for a in aliases:
                self.alias_to_canonical[a.lower()] = canonical
                self.exact_terms.add(a.lower())

    def match_medication(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Scans normalized text for recognized medication entities.
        Returns match dict with canonical name, raw matched string, and drug class, or None.
        """
        t_low = text.lower()
        
        # Sort terms by length descending to match multi-word phrases first (e.g. 'nicip plus' before 'nicip')
        sorted_terms = sorted(self.exact_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            # Word boundary matching
            pattern = r"\b" + re.escape(term) + r"\b"
            match = re.search(pattern, t_low)
            if match:
                canonical = self.alias_to_canonical.get(term, term)
                entry = self.entries.get(canonical)
                return {
                    "matched_text": match.group(0),
                    "canonical": canonical,
                    "drug_class": entry.drug_class if entry else "general",
                    "start": match.start(),
                    "end": match.end()
                }

        # Check generic regexes for medication stems (e.g. medc*, medic*)
        gen_match = re.search(r"\b(medc[a-z]*|medic[a-z]*|medec[a-z]*|medicat[a-z]*|meds?|tabs?|pills?)\b", t_low)
        if gen_match:
            return {
                "matched_text": gen_match.group(0),
                "canonical": "medication",
                "drug_class": "generic_term",
                "start": gen_match.start(),
                "end": gen_match.end()
            }

        return None

    def is_known_medication(self, term: str) -> bool:
        t = term.lower().strip()
        return t in self.exact_terms or t in self.alias_to_canonical
