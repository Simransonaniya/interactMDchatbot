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
"""

import pytest
import uuid
from ai_orchestrator import ai_orchestrator
from patient_state import PatientStateManager


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

    # Turn: Doctor asks for pain character
    res = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Can you describe what the pain feels like?",
        session_id=session_id
    )

    reply_low = res["reply"].lower()
    # Must NOT return generic fallback
    assert "haven't really noticed" not in reply_low
    assert "anything like that" not in reply_low
    # Must contain pressure / elephant / squeezing / crushing
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
    Expected:
    message 1 -> state updated
    message 2 -> retrieves message 1 state
    message 3 -> retrieves relevant previous facts
    message 4 -> remains consistent
    message 5 -> remains consistent
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

    # Message 2: Pain character inquiry (must retrieve/ground on elephant / pressure)
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
