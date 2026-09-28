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
Test 8 — Exact Production Reproduction Transcript
Test 9 — Intent Classification Unit Tests across all categories
Test 10 — Diagnosis Statements & Clinical Examination Requests
Test 11 — Breakfast & Meal History Handling (Known vs Unknown)
Test 12 — Full Production Conversation Sequence
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
    Must NOT repeat unrelated opening symptoms (dizziness/office) or return symptom fallback.
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
    assert "dizzy" not in reply_low
    assert "office" not in reply_low


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
        ("had you breakfast", "HISTORY_QUESTION", IntentCategory.SOCIAL_HISTORY),
        ("what was you eat in your breakfast", "HISTORY_QUESTION", IntentCategory.SOCIAL_HISTORY),
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


def test_11_breakfast_and_diet_handling():
    """
    Test 11: Food and breakfast inquiry handling.
    Must NOT return generic symptom fallback ("I haven't really noticed anything like that").
    Must communicate uncertainty/unknown properly.
    """
    session_id = f"test-diet-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Inquiries about breakfast
    r1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="had you breakfast",
        session_id=session_id
    )
    reply1 = r1["reply"].lower()
    assert "haven't really noticed anything like that" not in reply1
    assert any(k in reply1 for k in ["breakfast", "remember", "eat", "ate", "rushing", "office"])

    r2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="what was you eat in your breakfast",
        session_id=session_id
    )
    reply2 = r2["reply"].lower()
    assert "haven't really noticed anything like that" not in reply2
    assert any(k in reply2 for k in ["breakfast", "remember", "eat", "ate", "rushing", "office"])


def test_12_full_production_conversation_sequence():
    """
    Test 12: Full Production Conversation from User Prompt
    1. Patient: Initial statement
    2. Doctor: "Can you describe what the pain feels like?" -> Focused pain character
    3. Doctor: "Have you experienced any cold sweats, nausea, or vomiting?" -> Positive cold sweat
    4. Doctor: "you should take medicine home and take rest" -> Management/Medication ack
    5. Doctor: "you can take a Paracetamol if you feel like a fever" -> Medication statement ack
    6. Doctor: "do you have a PCOD also" -> Not applicable (male)
    7. Doctor: "had you breakfast" -> Controlled unknown (breakfast)
    8. Doctor: "what was you eat in your breakfast" -> Controlled unknown (breakfast)
    9. Doctor: "are you sure?" -> Clarification confirmation
    10. Doctor: "How severe is your discomfort?" (twice) -> 8/10 consistent
    """
    session_id = f"test-full-seq-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # 1. Pain character
    t1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Can you describe what the pain feels like?",
        session_id=session_id
    )
    rep1 = t1["reply"].lower()
    assert any(k in rep1 for k in ["pressure", "elephant", "crushing", "squeezing", "heavy", "chest"])
    assert "dizzy" not in rep1

    # 2. Associated symptoms
    t2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Have you experienced any cold sweats, nausea, or vomiting?",
        session_id=session_id
    )
    rep2 = t2["reply"].lower()
    assert any(k in rep2 for k in ["sweat", "yes"])

    # 3. Management statement
    t3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you should take medicine home and take rest",
        session_id=session_id
    )
    rep3 = t3["reply"].lower()
    assert "haven't really noticed" not in rep3
    assert any(k in rep3 for k in ["okay", "doctor", "rest", "relieve", "chest pain", "breathe"])

    # 4. Medication statement
    t4 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you can take a Paracetamol if you feel like a fever",
        session_id=session_id
    )
    rep4 = t4["reply"].lower()
    assert "haven't really noticed" not in rep4
    assert any(k in rep4 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "paracetamol"])

    # 5. PCOD (male)
    t5 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="do you have a PCOD also",
        session_id=session_id
    )
    rep5 = t5["reply"].lower()
    assert "male" in rep5 or "doesn't apply" in rep5

    # 6. Breakfast
    t6 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="had you breakfast",
        session_id=session_id
    )
    rep6 = t6["reply"].lower()
    assert "haven't really noticed anything like that" not in rep6
    assert any(k in rep6 for k in ["breakfast", "remember", "eat", "ate", "rushing", "office"])

    # 7. Breakfast details
    t7 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="what was you eat in your breakfast",
        session_id=session_id
    )
    rep7 = t7["reply"].lower()
    assert "haven't really noticed anything like that" not in rep7
    assert any(k in rep7 for k in ["breakfast", "remember", "eat", "ate", "rushing", "office"])

    # 8. Clarification
    t8 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="are you sure?",
        session_id=session_id
    )
    rep8 = t8["reply"].lower()
    assert "haven't really noticed" not in rep8
    assert any(k in rep8 for k in ["yes", "sure", "definitely", "feel"])

    # 9. Severity ask 1
    t9 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="How severe is your discomfort?",
        session_id=session_id
    )
    assert "8" in t9["reply"] or "eight" in t9["reply"].lower()

    # 10. Severity ask 2
    t10 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="How severe is your discomfort?",
        session_id=session_id
    )
    assert "8" in t10["reply"] or "eight" in t10["reply"].lower()


def test_13_advanced_intent_routing_and_temporal_diet():
    """
    Test 13: Regression Tests A, B, C, J for Temporal / Meal slot routing:
    - Test A: 'Had you breakfast?' -> breakfast slot
    - Test B: 'What did you eat yesterday for lunch and dinner?' -> slots = [lunch, dinner], time = previous_day
    - Test C: Turn 1 (Breakfast) then Turn 2 (Lunch/Dinner) must NOT reuse breakfast response on Turn 2!
    - Test J: Unknown diet query for dinner -> dinner specific response, not breakfast
    """
    session_id = f"test-temporal-diet-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Turn 1: Breakfast
    r1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="had you breakfast",
        session_id=session_id
    )
    rep1 = r1["reply"].lower()
    assert "haven't really noticed anything like that" not in rep1
    assert "breakfast" in rep1

    # Turn 2: Lunch & Dinner yesterday (must NOT return breakfast!)
    r2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="what did you eat tomorrow yesterday in your lunch and dinner",
        session_id=session_id
    )
    rep2 = r2["reply"].lower()
    assert "haven't really noticed anything like that" not in rep2
    assert "breakfast" not in rep2
    assert any(k in rep2 for k in ["lunch", "dinner"])
    assert "yesterday" in rep2

    # Turn 3: Dinner yesterday only
    r3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="What did you eat yesterday for dinner?",
        session_id=session_id
    )
    rep3 = r3["reply"].lower()
    assert "breakfast" not in rep3
    assert "dinner" in rep3


def test_14_medication_fragments_and_history():
    """
    Test 14: Regression Tests D, E, F, G, H, I:
    - Test D: 'What medicines do you currently take?' or 'had you eat any medicine' -> Amlodipine / Atorvastatin
    - Test E: 'you can take a nishchit plus medicine for your rest' -> MEDICATION_STATEMENT
    - Test F: 'niciplus' -> MEDICATION_NAME_FRAGMENT
    - Test G: 'paracetomol tablet' -> MEDICATION_NAME_FRAGMENT / MEDICATION_STATEMENT
    - Test H: 'Take some rest.' -> MANAGEMENT_STATEMENT
    - Test I: 'Do you have PCOD?' for male patient -> NOT_APPLICABLE
    """
    session_id = f"test-med-frag-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Test D: Current medications history
    r_med_hx = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="had you eat any medicine",
        session_id=session_id
    )
    rep_med_hx = r_med_hx["reply"].lower()
    assert any(k in rep_med_hx for k in ["amlodipine", "atorvastatin", "medications"])

    # Test E: Clinician Medication Statement with brand typo
    r_med_stmt = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="you can take a nishchit plus medicine for your rest",
        session_id=session_id
    )
    rep_med_stmt = r_med_stmt["reply"].lower()
    assert "haven't really noticed" not in rep_med_stmt
    assert any(k in rep_med_stmt for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "help"])

    # Test F: Medication Name Fragment (niciplus)
    r_frag1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="niciplus",
        session_id=session_id
    )
    rep_frag1 = r_frag1["reply"].lower()
    assert "haven't really noticed anything like that" not in rep_frag1
    assert any(k in rep_frag1 for k in ["medication", "medicine", "take", "doctor", "relieve", "chest pain", "breathe"])

    # Test G: Medication Name + Form with typo (paracetomol tablet)
    r_frag2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="paracetomol tablet",
        session_id=session_id
    )
    rep_frag2 = r_frag2["reply"].lower()
    assert "haven't really noticed anything like that" not in rep_frag2
    assert any(k in rep_frag2 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "tablet", "medicine"])

    # Test H: Management statement
    r_mgmt = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Take some rest.",
        session_id=session_id
    )
    rep_mgmt = r_mgmt["reply"].lower()
    assert "haven't really noticed" not in rep_mgmt
    assert any(k in rep_mgmt for k in ["rest", "sit down", "okay", "chest", "pressure"])

    # Test I: PCOD (male)
    r_pcod = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Do you have PCOD?",
        session_id=session_id
    )
    rep_pcod = r_pcod["reply"].lower()
    assert "male" in rep_pcod or "doesn't apply" in rep_pcod


def test_clarification_and_challenge_repair_conversation_flow():
    """
    Validates complete multi-turn dialogue with challenge, clarification, confirmation, and memory defense:
    1. Initial Complaint
    2. Timing: "When did this start and how long has it lasted?" -> ~45 minutes
    3. Breakfast: "Had you breakfast?" -> breakfast response
    4. Dinner: "What had you in dinner?" -> dinner response
    5. Ungrammatical Challenge: "how can you don't know" -> challenge response (NOT symptom fallback!)
    6. Confirmation: "Are you sure?" -> confirmation response (NOT symptom fallback!)
    7. New Question: "What did you eat for lunch?" -> lunch-specific response
    8. Challenge: "Why can't you remember?" -> challenge response (NOT symptom fallback!)
    9. Colloquial Challenge: "are you made you don't know anything" -> challenge response
    10. Severity Confirmation: "My discomfort is 8/10" -> "Are you sure?" -> 8/10 confirmation
    """
    session_id = f"test-flow-challenge-{uuid.uuid4()}"
    case_id = "chest_pain_001"

    # Step 1: Timing / Onset
    r1 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="When did this start and how long has it lasted?",
        session_id=session_id
    )
    rep1 = r1["reply"].lower()
    assert "45 minutes" in rep1
    assert "haven't really noticed anything like that" not in rep1

    # Step 2: Breakfast
    r2 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Had you breakfast?",
        session_id=session_id
    )
    rep2 = r2["reply"].lower()
    assert "breakfast" in rep2 or "rushing" in rep2
    assert "haven't really noticed anything like that" not in rep2

    # Step 3: Dinner
    r3 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="What had you in dinner?",
        session_id=session_id
    )
    rep3 = r3["reply"].lower()
    assert "dinner" in rep3 or "yesterday" in rep3 or "chest pain" in rep3
    assert "haven't really noticed anything like that" not in rep3

    # Step 4: Ungrammatical Challenge: "how can you don't know"
    r4 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="how can you don't know",
        session_id=session_id
    )
    rep4 = r4["reply"].lower()
    assert "haven't really noticed anything like that" not in rep4
    assert any(k in rep4 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])
    assert r4.get("intent_category") in ["CHALLENGE", "CLARIFICATION"] or r4.get("relationship") == "CHALLENGE"

    # Step 5: Confirmation: "Are you sure?"
    r5 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Are you sure?",
        session_id=session_id
    )
    rep5 = r5["reply"].lower()
    assert "haven't really noticed anything like that" not in rep5
    assert any(k in rep5 for k in ["sure", "remember", "chest pain", "focus", "dizziness", "yes"])

    # Step 6: New question: "What did you eat for lunch?"
    r6 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="What did you eat for lunch?",
        session_id=session_id
    )
    rep6 = r6["reply"].lower()
    assert "haven't really noticed anything like that" not in rep6
    assert any(k in rep6 for k in ["lunch", "remember", "eat", "pain"])

    # Step 7: Challenge: "Why can't you remember?"
    r7 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Why can't you remember?",
        session_id=session_id
    )
    rep7 = r7["reply"].lower()
    assert "haven't really noticed anything like that" not in rep7
    assert any(k in rep7 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])

    # Step 8: Ungrammatical challenge: "are you made you don't know anything"
    r8 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="are you made you don't know anything",
        session_id=session_id
    )
    rep8 = r8["reply"].lower()
    assert "haven't really noticed anything like that" not in rep8
    assert any(k in rep8 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])

    # Step 9: Severity question + Confirmation
    r9 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="On a scale of 1 to 10, how severe is your discomfort right now?",
        session_id=session_id
    )
    rep9 = r9["reply"].lower()
    assert "8" in rep9

    r10 = ai_orchestrator.process_turn_sync(
        case_id=case_id,
        user_message="Are you sure?",
        session_id=session_id
    )
    rep10 = r10["reply"].lower()
    assert "8" in rep10
    assert "haven't really noticed anything like that" not in rep10


