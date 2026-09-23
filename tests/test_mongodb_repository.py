"""
InteractMD — MongoDB Repository & Persistence Tests.
Verifies:
1. Case retrieval from MongoDB for all 3 clinical benchmark cases.
2. Session creation with version tracking.
3. Message persistence in chronological order.
4. Interaction event audit logging.
5. User data isolation.
"""

import pytest
import time
import uuid
from mongo_db import mongo_manager


def test_01_mongodb_connected():
    """Verify live connection to MongoDB Atlas."""
    assert mongo_manager.is_connected is True
    assert mongo_manager.db is not None


def test_02_all_three_cases_exist_in_mongodb():
    """Verify chest_pain_001, dyspnea_002, and abdominal_pain_003 are present in MongoDB."""
    cases = mongo_manager.get_all_cases()
    assert len(cases) >= 3

    case_ids = [c.get("case_id") or c.get("id") for c in cases]
    assert "chest_pain_001" in case_ids
    assert "dyspnea_002" in case_ids
    assert "abdominal_pain_003" in case_ids


def test_03_case_structure_and_tri_state_facts():
    """Verify structured clinical case schema and tri-state fact representation."""
    case = mongo_manager.get_case_by_id("chest_pain_001")
    assert case is not None
    assert case["title"] == "Acute Retrosternal Chest Pain"
    assert case["patient"]["name"] == "Robert Chen"
    assert case["patient"]["age"] == 58

    # Verify history facts
    history = case["history"]
    assert "chief_complaint" in history
    assert "associated_symptoms" in history
    assoc = history["associated_symptoms"]
    assert assoc["shortness_of_breath"]["value"] is True
    assert assoc["fever"]["value"] is False
    assert assoc["body_aches"]["value"] is None
    assert assoc["body_aches"]["state"] == "UNKNOWN"


def test_04_session_lifecycle_and_versioning():
    """Verify session creation, updating, and case versioning in MongoDB."""
    user_id = f"user-{uuid.uuid4().hex[:6]}"
    session_id = str(uuid.uuid4())

    session_doc = {
        "session_id": session_id,
        "id": session_id,
        "user_id": user_id,
        "case_id": "chest_pain_001",
        "case_version": 1,
        "status": "ACTIVE",
        "started_at": time.time(),
        "revealed_fact_ids": [],
        "completed_examinations": [],
        "ordered_investigations": []
    }
    mongo_manager.create_session(session_doc)

    # Retrieve and verify
    retrieved = mongo_manager.get_session(session_id)
    assert retrieved is not None
    assert retrieved["user_id"] == user_id
    assert retrieved["case_id"] == "chest_pain_001"
    assert retrieved["case_version"] == 1
    assert retrieved["status"] == "ACTIVE"

    # Update session
    mongo_manager.record_fact_revealed(session_id, "radiation")
    mongo_manager.record_examination_completed(session_id, "exam-cv", {"status": "done"})
    mongo_manager.record_investigation_ordered(session_id, "inv-ecg", {"status": "done"})

    updated = mongo_manager.get_session(session_id)
    assert "radiation" in updated["revealed_fact_ids"]
    assert "exam-cv" in updated["completed_examinations"]
    assert "inv-ecg" in updated["ordered_investigations"]


def test_05_message_persistence():
    """Verify chronological persistence of dialogue turns in MongoDB."""
    session_id = str(uuid.uuid4())
    t1 = time.time()
    t2 = t1 + 1.0

    mongo_manager.save_message({
        "session_id": session_id,
        "role": "learner",
        "content": "Where does it hurt?",
        "timestamp": t1
    })

    mongo_manager.save_message({
        "session_id": session_id,
        "role": "patient",
        "content": "Right in the middle of my chest.",
        "timestamp": t2
    })

    messages = mongo_manager.get_session_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Where does it hurt?"
    assert messages[1]["content"] == "Right in the middle of my chest."


def test_06_interaction_events_auditing():
    """Verify audit logging of interaction events."""
    session_id = str(uuid.uuid4())
    mongo_manager.save_event(session_id, "EXAM_PERFORMED", {"exam": "cardiovascular"})
    mongo_manager.save_event(session_id, "INVESTIGATION_ORDERED", {"test": "STAT 12-Lead ECG"})

    events = mongo_manager.get_session_events(session_id)
    assert len(events) == 2
    assert events[0]["event_type"] == "EXAM_PERFORMED"
    assert events[1]["event_type"] == "INVESTIGATION_ORDERED"


def test_07_user_data_isolation():
    """Verify User B cannot access User A's sessions in MongoDB queries."""
    user_a = f"user-a-{uuid.uuid4().hex[:6]}"
    user_b = f"user-b-{uuid.uuid4().hex[:6]}"

    s_a = str(uuid.uuid4())
    s_b = str(uuid.uuid4())

    mongo_manager.create_session({"session_id": s_a, "id": s_a, "user_id": user_a, "case_id": "chest_pain_001"})
    mongo_manager.create_session({"session_id": s_b, "id": s_b, "user_id": user_b, "case_id": "dyspnea_002"})

    sessions_a = mongo_manager.get_user_sessions(user_a)
    sessions_b = mongo_manager.get_user_sessions(user_b)

    assert any(s["session_id"] == s_a for s in sessions_a)
    assert not any(s["session_id"] == s_b for s in sessions_a)
    assert any(s["session_id"] == s_b for s in sessions_b)
    assert not any(s["session_id"] == s_a for s in sessions_b)
