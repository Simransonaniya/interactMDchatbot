"""Quick verification script for Patient Dialogue Rules."""
import sys
sys.path.insert(0, ".")

from ai_engine import AIPatientEngine

engine = AIPatientEngine()
case_id = "case-acs-1"
history = []

TESTS = [
    ("Greeting",                    "Hello doctor"),
    ("Open-ended question",         "What brings you here today?"),
    ("Onset (1st-person check)",    "When did this start?"),
    ("Radiation (his->my fix)",     "Does the pain radiate anywhere?"),
    ("Medical jargon",              "STEMI"),
    ("Quality of pain",             "Can you describe what the pain feels like?"),
    ("Severity 1-10",               "Rate pain 1 to 10"),
    ("Past medical history",        "Any past medical history?"),
    ("Fallback (no info leak)",     "zzz nonsense query xyz"),
    ("Pure empathy",                "Don't worry, I'm here to help you"),
    ("Social history",              "Do you smoke or drink?"),
    ("Medications",                 "What medications are you taking?"),
    ("Allergies",                   "Any allergies?"),
    ("Family history",              "Any family history of heart disease?"),
    ("Sweating",                    "Are you sweating?"),
    ("Nausea",                      "Do you feel nauseous?"),
]

import re as _re

def _has_violations(reply: str) -> list:
    """Word-boundary-aware violation detection."""
    r = reply.lower()
    hits = []
    if _re.search(r'\bhis\b', r):   hits.append("his")
    if _re.search(r'\bhe\b', r):    hits.append("he")
    if _re.search(r'\bshe\b', r):   hits.append("she")
    if "the patient" in r:          hits.append("the patient")
    if "facts[" in r:               hits.append("facts[")
    return hits

print("=" * 65)
print("   PATIENT DIALOGUE RULES — Verification Report")
print("=" * 65)

passed = 0
failed = 0

for label, query in TESTS:
    result = engine.process_turn(case_id, query, history)
    reply  = result["reply"]
    cat    = result["category"]

    violations = _has_violations(reply)
    status = "[PASS]" if not violations else f"[FAIL] (found: {violations})"
    if not violations:
        passed += 1
    else:
        failed += 1

    print(f"\n[{status}] {label}")
    print(f"  Q: {query}")
    print(f"  A: {reply[:130]}{'...' if len(reply) > 130 else ''}")
    print(f"  Category: {cat}")

print("\n" + "=" * 65)
print(f"  Results: {passed}/{len(TESTS)} passed, {failed} failed")
print("=" * 65)
