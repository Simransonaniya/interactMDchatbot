"""
InteractMD — Standardized Medication Terminology Service.
Integrates internal medication catalog with optional NLM RxNorm REST API lookups.
Features:
- Sub-millisecond in-memory LRU caching
- Configurable timeout (500ms default)
- Safe offline fallback (never blocks dialogue if API is unreachable)
- Source tracking (INTERNAL_CATALOG, RXNORM_API, CASE_SPECIFIC)
"""

import time
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from medical_nlu.medication_catalog import MedicationCatalog, MedicationConcept


@dataclass
class TerminologyLookupResult:
    query: str
    canonical_name: str
    display_name: str
    rxcui: Optional[str] = None
    drug_class: str = "general"
    active_ingredients: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)
    dosage_forms: List[str] = field(default_factory=list)
    common_strengths: List[str] = field(default_factory=list)
    indications_purpose: str = ""
    common_side_effects: List[str] = field(default_factory=list)
    is_combination: bool = False
    source: str = "INTERNAL_CATALOG"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "rxcui": self.rxcui,
            "drug_class": self.drug_class,
            "active_ingredients": self.active_ingredients,
            "brand_names": self.brand_names,
            "dosage_forms": self.dosage_forms,
            "common_strengths": self.common_strengths,
            "indications_purpose": self.indications_purpose,
            "common_side_effects": self.common_side_effects,
            "is_combination": self.is_combination,
            "source": self.source,
            "confidence": self.confidence
        }


class TerminologyService:
    _instance: Optional["TerminologyService"] = None

    def __init__(self, enable_rxnorm_api: bool = True, timeout_sec: float = 0.5):
        self.catalog = MedicationCatalog.get_instance()
        self.enable_rxnorm_api = enable_rxnorm_api
        self.timeout_sec = timeout_sec
        self.cache: Dict[str, TerminologyLookupResult] = {}
        self.rxnorm_base_url = "https://rxnav.nlm.nih.gov/REST"

    @classmethod
    def get_instance(cls) -> "TerminologyService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def lookup(self, term: str, case_context_meds: Optional[List[str]] = None) -> Optional[TerminologyLookupResult]:
        if not term or not term.strip():
            return None

        norm_term = term.lower().strip()

        # 1. Check in-memory fast cache
        if norm_term in self.cache:
            return self.cache[norm_term]

        # 2. Check internal medication catalog
        catalog_entry = self.catalog.lookup(norm_term)
        if catalog_entry:
            res = TerminologyLookupResult(
                query=term,
                canonical_name=catalog_entry.canonical_name,
                display_name=catalog_entry.display_name,
                drug_class=catalog_entry.drug_class,
                active_ingredients=catalog_entry.active_ingredients,
                brand_names=catalog_entry.brand_names,
                dosage_forms=catalog_entry.dosage_forms,
                common_strengths=catalog_entry.common_strengths,
                indications_purpose=catalog_entry.indications_purpose,
                common_side_effects=catalog_entry.common_side_effects,
                is_combination=catalog_entry.is_combination,
                source=catalog_entry.source,
                confidence=0.99
            )
            self.cache[norm_term] = res
            return res

        # 3. Check case-specific medications if provided
        if case_context_meds:
            for cm in case_context_meds:
                if norm_term in cm.lower():
                    res = TerminologyLookupResult(
                        query=term,
                        canonical_name=norm_term,
                        display_name=cm,
                        drug_class="case_fact",
                        active_ingredients=[norm_term],
                        source="CASE_FACT",
                        confidence=0.95
                    )
                    self.cache[norm_term] = res
                    return res

        # 4. Optional RxNorm REST API lookup (with safe timeout & offline catch)
        if self.enable_rxnorm_api:
            rx_res = self._lookup_rxnorm_api(norm_term)
            if rx_res:
                self.cache[norm_term] = rx_res
                return rx_res

        return None

    def _lookup_rxnorm_api(self, term: str) -> Optional[TerminologyLookupResult]:
        try:
            encoded = urllib.parse.quote(term)
            url = f"{self.rxnorm_base_url}/rxcui.json?name={encoded}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "InteractMD-ClinicalSimulation/2.0", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    id_group = data.get("idGroup", {})
                    rxnorm_ids = id_group.get("rxnormId", [])
                    if rxnorm_ids:
                        rxcui = rxnorm_ids[0]
                        res = TerminologyLookupResult(
                            query=term,
                            canonical_name=term,
                            display_name=term.title(),
                            rxcui=rxcui,
                            drug_class="rxnorm_normalized",
                            active_ingredients=[term],
                            source="RXNORM_API",
                            confidence=0.92
                        )
                        return res
        except Exception:
            # Safe silent fallback on network timeout or offline state
            pass

        return None
