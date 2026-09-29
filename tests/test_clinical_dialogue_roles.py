"""
InteractMD — Regression Tests for Message Role Classification and Semantic Grounding.
Tests all bug scenarios specified in the Clinical Dialogue Understanding and Patient Response requirements.
"""

import pytest
from question_classifier import QuestionClassifier, ClassifiedIntent, IntentCategory
from fact_retriever import FactRetriever, RetrievedFact
from patient_state import PatientSimulationState, FactState
from medical_nlu.intent_classifier import MedicalIntentClassifier, MedicalIntent


@pytest.fixture
def robert_chen_case():
    return {
        "case_id": "robert_chen_ami",
        "title": "Acute Myocardial Infarction — Robert Chen",
        "patient": {
            "name": "Robert Chen",
            "age": 54,
            "gender": "male"
        },
        "history": {
            "chief_complaint": "Chest pain, dizziness, and cold sweats for 45 minutes.",
            "character": "It feels like a heavy crushing pressure, almost like an elephant is sitting right in the middle of my chest.",
            "location": "Right in the middle of my chest.",
            "radiation": "Yes, it radiates up into my jaw and down my left arm.",
            "severity": "About an 8 out of 10 right now.",
            "onset": "About 45 minutes ago while driving into the office.",
            "timing": "Started suddenly and has been continuous and worsening.",
            "associated_symptoms": {
                "shortness_of_breath": {"value": True},
                "sweating": {"value": True},
                "dizziness": {"value": True},
                "nausea": {"value": True}
            },
            "past_medical_history": ["Hypertension for 6 years", "Hyperlipidemia for 6 years"],
            "medications": ["Amlodipine 5 mg daily", "Atorvastatin 20 mg daily"],
            "allergies": ["No known drug allergies (NKDA)"],
            "social_history": "Smokes 0.5 packs per day for 25 years. Drinks 1-2 glasses of wine on weekends. No illicit substances.",
            "family_history": "Father had a fatal myocardial infarction at age 58."
        }
    }


def test_bug_management_relaxation(robert_chen_case):
    """
    Bug 2: Doctor says 'deep breath relaxation for 5 to 10 minutes'.
    Expected: Classifies as MANAGEMENT_INSTRUCTION, responds acknowledging breathing/relaxation,
    does NOT return 'I didn't quite catch what you said' or UNKNOWN_SYMPTOM or CURRENT_MEDICATIONS.
    """
    msg = "deep breath relaxation for 5 to 10 minutes"
    intent = QuestionClassifier.classify(msg)

    assert intent.category in [IntentCategory.MANAGEMENT_INSTRUCTION, IntentCategory.MANAGEMENT_STATEMENT]
    assert intent.primary_type == "MANAGEMENT_INSTRUCTION"
    assert intent.slots == []

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "deep breath" in fact.permitted_statement.lower() or "relax" in fact.permitted_statement.lower()
    assert "didn't quite catch" not in fact.permitted_statement.lower()
    assert "amlodipine" not in fact.permitted_statement.lower()
    assert "smoke" not in fact.permitted_statement.lower()


def test_bug_lifestyle_management(robert_chen_case):
    """
    Bug 3: Doctor says 'regular means and sleep reduce caffeine energy drink and lightweight or exercise'.
    Expected: Classifies as LIFESTYLE_MANAGEMENT, acknowledges lifestyle advice,
    does NOT retrieve smoking/alcohol social history.
    """
    msg = "regular means and sleep reduce caffeine energy drink and lightweight or exercise"
    intent = QuestionClassifier.classify(msg)

    assert intent.category == IntentCategory.LIFESTYLE_MANAGEMENT
    assert intent.primary_type == "LIFESTYLE_MANAGEMENT"
    assert intent.slots == []

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "caffeine" in fact.permitted_statement.lower() or "sleep" in fact.permitted_statement.lower() or "exercise" in fact.permitted_statement.lower()
    # Critical rule: Must NOT dump smoking / alcohol history!
    assert "smoke" not in fact.permitted_statement.lower()
    assert "wine" not in fact.permitted_statement.lower()
    assert "pack" not in fact.permitted_statement.lower()
    assert "amlodipine" not in fact.permitted_statement.lower()


def test_bug_medication_statement_disprin(robert_chen_case):
    """
    Bug 4: Doctor says 'you can take a Disprin tablet'.
    Expected: Classifies as MEDICATION_STATEMENT, extracted medication entity = Disprin / aspirin,
    acknowledges taking the recommended medication, does NOT dump current medications (Amlodipine/Atorvastatin).
    """
    msg = "you can take a Disprin tablet"
    intent = QuestionClassifier.classify(msg)

    assert intent.category == IntentCategory.MEDICATION_STATEMENT
    assert intent.primary_type == "MEDICATION_STATEMENT"
    assert intent.slots == []

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "disprin" in fact.permitted_statement.lower() or "take" in fact.permitted_statement.lower()
    # Must NOT return current medications list
    assert "amlodipine" not in fact.permitted_statement.lower()
    assert "atorvastatin" not in fact.permitted_statement.lower()


def test_bug_clinician_claim_high_volume_medicine(robert_chen_case):
    """
    Bug 5: Doctor says 'you take in high volume of medicine that's why you get a anxiety'.
    Expected: Classifies as CLINICAL_CLAIM, reacts without converting claim to confirmed fact or dumping meds.
    """
    msg = "you take in high volume of medicine that's why you get a anxiety"
    intent = QuestionClassifier.classify(msg)

    assert intent.category in [IntentCategory.CLINICAL_CLAIM, IntentCategory.CLINICAL_INTERPRETATION]
    assert intent.primary_type == "CLINICAL_CLAIM"
    assert intent.slots == []

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "didn't quite catch" not in fact.permitted_statement.lower()
    # Must NOT treat as "What medications do you take?"
    assert not fact.permitted_statement.startswith("I take my daily medications: Amlodipine")


def test_bug_clinician_claim_empty_stomach(robert_chen_case):
    """
    Bug 6: Doctor says 'maybe you take your medicine with empty stomach'.
    Expected: Classifies as CLINICAL_CLAIM, reacts with uncertainty without inventing causality.
    """
    msg = "maybe you take your medicine with empty stomach"
    intent = QuestionClassifier.classify(msg)

    assert intent.category in [IntentCategory.CLINICAL_CLAIM, IntentCategory.CLINICAL_INTERPRETATION]
    assert intent.primary_type == "CLINICAL_CLAIM"
    assert intent.slots == []

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert not fact.permitted_statement.startswith("I take my daily medications: Amlodipine")


def test_bug_contextual_medication_breakfast_question(robert_chen_case):
    """
    Bug 7: Doctor asks 'before taking the medicine had you breakfast'.
    Expected: Classifies as CONTEXTUAL_HISTORY_QUESTION with medication + diet references.
    """
    msg = "before taking the medicine had you breakfast"
    intent = QuestionClassifier.classify(msg)

    assert intent.category == IntentCategory.CONTEXTUAL_HISTORY_QUESTION
    assert intent.primary_type == "CONTEXTUAL_HISTORY_QUESTION"
    assert intent.medication_reference is True
    assert intent.diet_reference is True
    assert intent.meal == "breakfast"
    assert intent.temporal_relation == "before_medication"

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "water" in fact.permitted_statement.lower() or "breakfast" in fact.permitted_statement.lower()


def test_real_medication_history_question(robert_chen_case):
    """
    Genuine question 'What medications do you take?' MUST retrieve current medications.
    """
    msg = "What medications do you take?"
    intent = QuestionClassifier.classify(msg)

    assert intent.category in [IntentCategory.MEDICATIONS, IntentCategory.MEDICATION_HISTORY]
    assert intent.primary_type == "HISTORY_QUESTION"

    fact = FactRetriever.retrieve(intent, robert_chen_case)
    assert fact.state == FactState.AVAILABLE
    assert "amlodipine" in fact.permitted_statement.lower()
    assert "atorvastatin" in fact.permitted_statement.lower()


def test_full_robert_chen_regression_dialogue_sequence(robert_chen_case):
    """
    Section 25: Full Robert Chen regression dialogue turn-by-turn verification.
    """
    # Turn 1: Pain character
    t1_msg = "Can you describe what the pain feels like?"
    t1_intent = QuestionClassifier.classify(t1_msg)
    assert t1_intent.category == IntentCategory.CHARACTER
    t1_fact = FactRetriever.retrieve(t1_intent, robert_chen_case)
    assert "elephant" in t1_fact.permitted_statement.lower() or "crushing" in t1_fact.permitted_statement.lower()

    # Turn 2: Management / Relaxation
    t2_msg = "deep breath relaxation for 5 to 10 minutes"
    t2_intent = QuestionClassifier.classify(t2_msg)
    assert t2_intent.primary_type == "MANAGEMENT_INSTRUCTION"
    t2_fact = FactRetriever.retrieve(t2_intent, robert_chen_case)
    assert "deep breath" in t2_fact.permitted_statement.lower() or "relax" in t2_fact.permitted_statement.lower()

    # Turn 3: Lifestyle advice
    t3_msg = "reduce caffeine, get regular sleep, and do light exercise"
    t3_intent = QuestionClassifier.classify(t3_msg)
    assert t3_intent.primary_type == "LIFESTYLE_MANAGEMENT"
    t3_fact = FactRetriever.retrieve(t3_intent, robert_chen_case)
    assert "caffeine" in t3_fact.permitted_statement.lower() or "sleep" in t3_fact.permitted_statement.lower()
    assert "smoke" not in t3_fact.permitted_statement.lower()

    # Turn 4: Medication statement
    t4_msg = "you can take a Disprin tablet"
    t4_intent = QuestionClassifier.classify(t4_msg)
    assert t4_intent.primary_type == "MEDICATION_STATEMENT"
    t4_fact = FactRetriever.retrieve(t4_intent, robert_chen_case)
    assert "disprin" in t4_fact.permitted_statement.lower()
    assert "amlodipine" not in t4_fact.permitted_statement.lower()

    # Turn 5: Clinical claim - high volume medicine
    t5_msg = "you take in high volume of medicine that's why you get anxiety"
    t5_intent = QuestionClassifier.classify(t5_msg)
    assert t5_intent.primary_type == "CLINICAL_CLAIM"
    t5_fact = FactRetriever.retrieve(t5_intent, robert_chen_case)
    assert not t5_fact.permitted_statement.startswith("I take my daily medications: Amlodipine")

    # Turn 6: Clinical claim - empty stomach
    t6_msg = "maybe you take your medicine with an empty stomach"
    t6_intent = QuestionClassifier.classify(t6_msg)
    assert t6_intent.primary_type == "CLINICAL_CLAIM"
    t6_fact = FactRetriever.retrieve(t6_intent, robert_chen_case)
    assert not t6_fact.permitted_statement.startswith("I take my daily medications: Amlodipine")

    # Turn 7: Contextual medication + breakfast
    t7_msg = "before taking the medicine had you breakfast"
    t7_intent = QuestionClassifier.classify(t7_msg)
    assert t7_intent.primary_type == "CONTEXTUAL_HISTORY_QUESTION"
    t7_fact = FactRetriever.retrieve(t7_intent, robert_chen_case)
    assert "water" in t7_fact.permitted_statement.lower() or "breakfast" in t7_fact.permitted_statement.lower()
