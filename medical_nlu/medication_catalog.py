"""
InteractMD — Medical Medication Catalog & Knowledge Repository.
Maintains structured drug concepts, ingredients, brand mappings, combination medications,
dosages, routes, indications, and side effects.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class MedicationConcept:
    canonical_name: str
    display_name: str
    drug_class: str
    active_ingredients: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    common_misspellings: List[str] = field(default_factory=list)
    dosage_forms: List[str] = field(default_factory=list)
    default_routes: List[str] = field(default_factory=list)
    common_strengths: List[str] = field(default_factory=list)
    indications_purpose: str = ""
    common_side_effects: List[str] = field(default_factory=list)
    source: str = "INTERNAL_CATALOG"

    @property
    def is_combination(self) -> bool:
        return len(self.active_ingredients) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "drug_class": self.drug_class,
            "active_ingredients": self.active_ingredients,
            "brand_names": self.brand_names,
            "aliases": self.aliases,
            "common_misspellings": self.common_misspellings,
            "dosage_forms": self.dosage_forms,
            "default_routes": self.default_routes,
            "common_strengths": self.common_strengths,
            "indications_purpose": self.indications_purpose,
            "common_side_effects": self.common_side_effects,
            "is_combination": self.is_combination,
            "source": self.source
        }


class MedicationCatalog:
    _instance: Optional["MedicationCatalog"] = None

    def __init__(self, json_path: Optional[str] = None):
        self.concepts: Dict[str, MedicationConcept] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.exact_terms: Set[str] = set()
        self.ingredient_to_canonicals: Dict[str, Set[str]] = {}

        if json_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(base_dir, "data", "lexicons", "medications.json")

        self.load(json_path)

    @classmethod
    def get_instance(cls) -> "MedicationCatalog":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            self._load_defaults()
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, val in data.items():
                canonical = val.get("canonical", key)
                display = val.get("display_name", canonical.title())
                drug_class = val.get("class", "general")
                ingredients = val.get("active_ingredients", [canonical])
                brands = val.get("brand_names", [])
                aliases = val.get("aliases", [canonical])
                misspellings = val.get("common_misspellings", [])
                forms = val.get("dosage_forms", ["tablet"])
                routes = val.get("default_routes", ["oral"])
                strengths = val.get("common_strengths", [])
                purpose = val.get("indications_purpose", "")
                side_effects = val.get("common_side_effects", [])

                concept = MedicationConcept(
                    canonical_name=canonical,
                    display_name=display,
                    drug_class=drug_class,
                    active_ingredients=ingredients,
                    brand_names=brands,
                    aliases=aliases,
                    common_misspellings=misspellings,
                    dosage_forms=forms,
                    default_routes=routes,
                    common_strengths=strengths,
                    indications_purpose=purpose,
                    common_side_effects=side_effects,
                    source="INTERNAL_CATALOG"
                )
                self.concepts[canonical] = concept

                # Index aliases
                for alias in aliases:
                    norm = alias.lower().strip()
                    self.alias_to_canonical[norm] = canonical
                    self.exact_terms.add(norm)

                # Index brand names
                for brand in brands:
                    norm = brand.lower().strip()
                    self.alias_to_canonical[norm] = canonical
                    self.exact_terms.add(norm)

                # Index misspellings
                for misspelling in misspellings:
                    norm = misspelling.lower().strip()
                    self.alias_to_canonical[norm] = canonical
                    self.exact_terms.add(norm)

                # Index active ingredients
                for ing in ingredients:
                    ing_norm = ing.lower().strip()
                    if ing_norm not in self.ingredient_to_canonicals:
                        self.ingredient_to_canonicals[ing_norm] = set()
                    self.ingredient_to_canonicals[ing_norm].add(canonical)

        except Exception as e:
            print(f"[MedicationCatalog Error] Failed to load catalog from {file_path}: {e}")
            self._load_defaults()

    def _load_defaults(self):
        defaults = [
            ("amlodipine", "Amlodipine", "calcium_channel_blocker", ["amlodipine"], ["Norvasc"], ["amlodipine", "norvasc", "amlodipin"], ["amlodepine"]),
            ("atorvastatin", "Atorvastatin", "statin", ["atorvastatin"], ["Lipitor"], ["atorvastatin", "lipitor", "atorva"], ["atorvastin"]),
            ("paracetamol", "Paracetamol", "analgesic", ["paracetamol"], ["Tylenol", "Panadol", "Crocin", "Dolo 650"], ["paracetamol", "acetaminophen", "tylenol", "crocin", "dolo"], ["paracetomol", "paracitamol"]),
            ("nicip plus", "Nicip Plus", "combination_nsaid", ["nimesulide", "paracetamol"], ["Nicip Plus"], ["nicip plus", "nicip", "niciplus"], ["niciplus"]),
            ("albuterol", "Albuterol / Salbutamol", "bronchodilator", ["albuterol"], ["Ventolin", "ProAir"], ["albuterol", "salbutamol", "inhaler", "blue inhaler", "ventolin"], ["inhailer"]),
            ("aspirin", "Aspirin", "antiplatelet", ["aspirin"], ["Ecosprin", "Disprin", "Bayer"], ["aspirin", "asprin", "ecosprin", "disprin"], ["asprin"]),
            ("nitroglycerin", "Nitroglycerin", "nitrate", ["nitroglycerin"], ["Nitrostat"], ["nitroglycerin", "nitro", "sublingual nitroglycerin", "gtn"], ["nitroglycerine"]),
            ("sertraline", "Sertraline", "ssri", ["sertraline"], ["Zoloft"], ["sertraline", "zoloft", "sertra"], ["sertrakine"]),
            ("escitalopram", "Escitalopram", "ssri", ["escitalopram"], ["Lexapro", "Nexito"], ["escitalopram", "lexapro", "nexito"], ["escita;pram"]),
            ("medicine", "Medication", "generic_term", [], [], ["medicine", "medicines", "medication", "medications", "meds", "tablet", "pill"], ["medcian", "medicin"]),
        ]
        for canon, disp, dclass, ings, brands, aliases, miss in defaults:
            concept = MedicationConcept(
                canonical_name=canon,
                display_name=disp,
                drug_class=dclass,
                active_ingredients=ings,
                brand_names=brands,
                aliases=aliases,
                common_misspellings=miss,
                source="INTERNAL_DEFAULT"
            )
            self.concepts[canon] = concept
            for a in aliases + brands + miss:
                self.alias_to_canonical[a.lower().strip()] = canon
                self.exact_terms.add(a.lower().strip())

    def lookup(self, name_or_alias: str) -> Optional[MedicationConcept]:
        norm = name_or_alias.lower().strip()
        canonical = self.alias_to_canonical.get(norm)
        if canonical:
            return self.concepts.get(canonical)
        return self.concepts.get(norm)

    def register_case_medications(self, case_med_strings: List[str]):
        """Dynamically registers case-specific medications into the active runtime catalog."""
        for med_str in case_med_strings:
            if not med_str or not isinstance(med_str, str):
                continue
            cleaned = re.sub(r"\s*\([^)]*\)", "", med_str).strip()
            # Extract name part before dosage
            name_part = re.split(r"\s+\d+", cleaned)[0].strip().lower()
            if name_part and name_part not in self.alias_to_canonical:
                canonical = name_part
                concept = MedicationConcept(
                    canonical_name=canonical,
                    display_name=cleaned,
                    drug_class="case_defined",
                    active_ingredients=[canonical],
                    brand_names=[],
                    aliases=[canonical, cleaned.lower()],
                    common_misspellings=[],
                    source="CASE_SPECIFIC"
                )
                self.concepts[canonical] = concept
                self.alias_to_canonical[canonical] = canonical
                self.alias_to_canonical[cleaned.lower()] = canonical
                self.exact_terms.add(canonical)
                self.exact_terms.add(cleaned.lower())
