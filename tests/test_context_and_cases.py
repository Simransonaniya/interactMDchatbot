"""
InteractMD — AI Patient Context Management & Clinical Grounding Verification Suite.
Automated tests for:
Test 1 — Pain character consistency
Test 2 — Severity
Test 3 — Gender relevance
Test 4 — Repeated question consistency
Test 5 — Medication / treatment statement handling
Test 6 — Unknown information handling
Test 7 — Multi-turn conversation continuity (5+ turns)
Test 8 — Exact Current Production Reproduction Transcript
Test 9 — Intent Classification Unit Tests across all categories
Test 10 — Diagnosis Statements & Clinical Examination Requests
"""

import pytest
import uuid
from ai_orchestrator import ai_orchestrator
from patient_state import PatientStateManager
from question_classifier import QuestionClassifier, IntentCategory


def test_01_pain_character_consistency():
    """
    Test 1: Pain character
    Patient initial statement: 'Doctor, please... It feels like an elephant is sitting right in the middle of my chest.'
    Doctor: 'Can you describe what the pain feels like?'
    Expected:
    Response must remain consistent with chest pressure/heaviness and elephant on chest.
    Must NOT return 'I haven't really noticed anything like that, doctor.'
    """
    session_id = f"test-char-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    res = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Can you describe what the pain feels like?",
        session_id=session_id
    )

    reply_low = res["reply"].lower()
    assert "haven't really noticed" not in reply_low
    assert "anything like that" not in reply_low
    assert any(k in reply_low for k in ["pressure", "elephant", "crushing", "squeezing", "heavy", "chest"])


def test_02_severity():
    """
    Test 2: Severity
    Patient has: 8/10
    Doctor: 'How severe is the discomfort?'
    Expected:
    Response is consistent with approximately 8/10.
    """
    session_id = f"test-sev-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    res = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="How severe is the discomfort?",
        session_id=session_id
    )

    reply_low = res["reply"].lower()
    assert "haven't really noticed" not in reply_low
    assert "8" in res["reply"] or "eight" in reply_low or "severe" in reply_low


def test_03_gender_relevance():
    """
    Test 3: Gender relevance
    Male patient (Robert Chen):
    Doctor: 'Do you have PCOD?' or 'Do you have PCOS?'
    Expected:
    A contextually appropriate response explaining that the condition does not apply,
    rather than a generic symptom fallback.
    """
    session_id = f"test-gen-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    res = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Do you have PCOD?",
        session_id=session_id
    )

    reply_low = res["reply"].lower()
    assert "haven't really noticed" not in reply_low
    assert any(k in reply_low for k in ["male", "man", "doesn't apply", "not apply", "don't have pcod"])


def test_04_repeated_question_consistency():
    """
    Test 4: Repeated question
    Ask the same history question twice.
    The second answer must remain consistent and must not contradict the first answer.
    """
    session_id = f"test-rep-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # First ask
    r1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Can you describe what the pain feels like?",
        session_id=session_id
    )

    # Intervening question
    r_int = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Are you breaking out in a sweat?",
        session_id=session_id
    )
    assert "yes" in r_int["reply"].lower() or "sweat" in r_int["reply"].lower()

    # Second ask of the same question
    r2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="What does the chest pain feel like?",
        session_id=session_id
    )

    # Both answers must be consistent with heavy chest pressure / elephant
    for r in [r1, r2]:
        reply_low = r["reply"].lower()
        assert "haven't really noticed" not in reply_low
        assert any(k in reply_low for k in ["pressure", "elephant", "crushing", "squeezing", "heavy", "chest"])


def test_05_medication_statement_handling():
    """
    Test 5: Medication statement
    Doctor sends a medication / treatment statement:
    'you cab take sertrakine', 'you can take escita;pram', 'you can take this medician sertraline'
    Expected:
    The system must use the simulation's defined treatment-response policy rather than the generic symptom fallback.
    """
    session_id = f"test-med-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Test 5a: Misspelled SSRI instruction
    res1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you cab take sertrakine",
        session_id=session_id
    )
    reply1 = res1["reply"].lower()
    assert "haven't really noticed" not in reply1
    assert any(k in reply1 for k in ["prescribe", "chest pain", "pressure", "doctor", "help", "dizziness"])

    # Test 5b: Second medication instruction
    res2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you can take escita;pram",
        session_id=session_id
    )
    reply2 = res2["reply"].lower()
    assert "haven't really noticed" not in reply2
    assert any(k in reply2 for k in ["prescribe", "chest pain", "pressure", "doctor", "help", "dizziness"])

    # Test 5c: Standard medication instruction
    res3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you can take this medician sertraline",
        session_id=session_id
    )
    reply3 = res3["reply"].lower()
    assert "haven't really noticed" not in reply3


def test_06_unknown_information_handling():
    """
    Test 6: Unknown information
    Ask about a fact that does not exist in the case (e.g., earache, vitiligo, food yesterday).
    Expected:
    The patient should indicate uncertainty/absence according to the simulation rules.
    The model must NOT invent the answer.
    """
    session_id = f"test-unk-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    res = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Do you have an earache or ringing in your ears?",
        session_id=session_id
    )

    reply_low = res["reply"].lower()
    assert "no" in reply_low or "haven't noticed" in reply_low
    # Ensure no invented ear conditions
    assert "tinnitus" not in reply_low
    assert "otitis" not in reply_low


def test_07_conversation_continuity_multiturn():
    """
    Test 7: Conversation continuity
    Send at least 5 sequential messages and verify state is preserved and consistent across turns.
    """
    session_id = f"test-multi-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Message 1: Opening inquiry
    m1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Hello Mr. Chen, what brings you into the emergency room?",
        session_id=session_id
    )
    assert any(k in m1["reply"].lower() for k in ["pressure", "elephant", "chest", "discomfort"])

    # Message 2: Pain character inquiry
    m2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Can you describe what the pain feels like?",
        session_id=session_id
    )
    assert any(k in m2["reply"].lower() for k in ["pressure", "elephant", "crushing", "squeezing", "heavy"])

    # Message 3: Radiation inquiry
    m3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Does this discomfort spread anywhere else?",
        session_id=session_id
    )
    assert any(k in m3["reply"].lower() for k in ["jaw", "arm", "left"])

    # Message 4: Onset timing inquiry
    m4 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="When did this start happening?",
        session_id=session_id
    )
    assert any(k in m4["reply"].lower() for k in ["45 minutes", "minutes ago", "started"])

    # Message 5: Associated symptoms inquiry
    m5 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Are you feeling dizzy or breaking out in a sweat?",
        session_id=session_id
    )
    assert any(k in m5["reply"].lower() for k in ["yes", "dizzy", "sweat"])


def test_08_exact_production_reproduction_transcript():
    """
    Test 8: Exact Current Production Reproduction Sequence
    1. Doctor: "On a scale of 1 to 10, how severe is your discomfort right now?" -> 8/10
    2. Doctor: "On a scale of 1 to 10, how severe is your discomfort right now?" -> 8/10 (consistent)
    3. Doctor: "are you sure?" -> Confirms 8/10 (Must NOT return generic fallback)
    4. Doctor: "you should take rest" -> Management acknowledgment (Must NOT return symptom fallback)
    5. Doctor: "you should take tablet" -> Medication acknowledgment (Must NOT return symptom fallback)
    6. Doctor: "take paracetomol" -> Medication acknowledgment (Must NOT return symptom fallback)
    """
    session_id = f"test-reprod-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Step 1: Severity question
    t1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="On a scale of 1 to 10, how severe is your discomfort right now?",
        session_id=session_id
    )
    assert "8" in t1["reply"] or "eight" in t1["reply"].lower()
    assert "haven't really noticed" not in t1["reply"].lower()

    # Step 2: Repeated severity question
    t2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="On a scale of 1 to 10, how severe is your discomfort right now?",
        session_id=session_id
    )
    assert "8" in t2["reply"] or "eight" in t2["reply"].lower()
    assert "haven't really noticed" not in t2["reply"].lower()

    # Step 3: Clarification: "are you sure?"
    t3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="are you sure?",
        session_id=session_id
    )
    reply3 = t3["reply"].lower()
    assert "haven't really noticed" not in reply3
    assert "anything like that" not in reply3
    assert any(k in reply3 for k in ["yes", "sure", "8", "intense", "severe"])

    # Step 4: Management: "you should take rest"
    t4 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you should take rest",
        session_id=session_id
    )
    reply4 = t4["reply"].lower()
    assert "haven't really noticed" not in reply4
    assert "anything like that" not in reply4
    assert any(k in reply4 for k in ["rest", "sit down", "okay", "ease", "help"])

    # Step 5: Treatment: "you should take tablet"
    t5 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you should take tablet",
        session_id=session_id
    )
    reply5 = t5["reply"].lower()
    assert "haven't really noticed" not in reply5
    assert "anything like that" not in reply5
    assert any(k in reply5 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "tablet", "medicine"])

    # Step 6: Medication with typo: "take paracetomol"
    t6 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="take paracetomol",
        session_id=session_id
    )
    reply6 = t6["reply"].lower()
    assert "haven't really noticed" not in reply6
    assert "anything like that" not in reply6
    assert any(k in reply6 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "paracetamol", "paracetomol"])


def test_09_intent_classification_categories():
    """
    Test 9: Test that QuestionClassifier correctly categorizes distinct clinician intents.
    """
    cases = [
        ("How severe is your pain?", "HISTORY_QUESTION", IntentCategory.SEVERITY),
        ("When did this start?", "HISTORY_QUESTION", IntentCategory.ONSET_TIMING),
        ("Does the pain spread anywhere?", "HISTORY_QUESTION", IntentCategory.RADIATION),
        ("What does the discomfort feel like?", "HISTORY_QUESTION", IntentCategory.CHARACTER),
        ("Are you sure?", "CLARIFICATION", IntentCategory.CLARIFICATION),
        ("Really?", "CLARIFICATION", IntentCategory.CLARIFICATION),
        ("Is that correct?", "CONFIRMATION", IntentCategory.CONFIRMATION),
        ("So it's been going on for 45 minutes?", "CONFIRMATION", IntentCategory.CONFIRMATION),
        ("Take a slow breath, you're going to be okay.", "EMPATHY_REASSURANCE", IntentCategory.EMPATHY_REASSURANCE),
        ("You should take rest.", "MANAGEMENT_STATEMENT", IntentCategory.MANAGEMENT_STATEMENT),
        ("Let's have you sit down and rest.", "MANAGEMENT_STATEMENT", IntentCategory.MANAGEMENT_STATEMENT),
        ("Take this tablet.", "MEDICATION_STATEMENT", IntentCategory.MEDICATION_STATEMENT),
        ("Take paracetamol.", "MEDICATION_STATEMENT", IntentCategory.MEDICATION_STATEMENT),
        ("take paracetomol", "MEDICATION_STATEMENT", IntentCategory.MEDICATION_STATEMENT),
        ("I think this may be a heart attack.", "DIAGNOSIS_STATEMENT", IntentCategory.DIAGNOSIS_STATEMENT),
        ("This appears to be cardiac.", "DIAGNOSIS_STATEMENT", IntentCategory.DIAGNOSIS_STATEMENT),
        ("Let me examine your chest.", "EXAM_REQUEST", IntentCategory.EXAMINATION_REQUEST),
        ("Let's order an ECG.", "INVESTIGATION_REQUEST", IntentCategory.INVESTIGATION_REQUEST),
        ("What is your favorite movie?", "OFF_TOPIC", IntentCategory.OFF_TOPIC),
        ("...", "UNKNOWN", IntentCategory.UNCLEAR),
    ]

    for query, expected_primary, expected_cat in cases:
        classified = QuestionClassifier.classify(query)
        assert classified.primary_type == expected_primary, f"Query '{query}' primary type expected {expected_primary} got {classified.primary_type}"
        assert classified.category == expected_cat, f"Query '{query}' category expected {expected_cat} got {classified.category}"


def test_10_diagnosis_and_examination_interactions():
    """
    Test 10: Clinician gives a diagnosis or requests an exam.
    """
    session_id = f"test-dx-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Doctor gives a provisional diagnosis
    res_dx = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="I think this is a heart attack.",
        session_id=session_id
    )
    reply_dx = res_dx["reply"].lower()
    assert "heart attack" in reply_dx or "god" in reply_dx or "help" in reply_dx
    assert "haven't really noticed" not in reply_dx

    # Doctor performs examination
    res_exam = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Let me listen to your chest.",
        session_id=session_id
    )
    reply_exam = res_exam["reply"].lower()
    assert "sure" in reply_exam or "ahead" in reply_exam or "okay" in reply_exam
    assert "haven't really noticed" not in reply_exam
