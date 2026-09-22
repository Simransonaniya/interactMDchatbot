"""
Comprehensive Test Suite for InteractMD PostgreSQL Backend.
Verifies:
1. Fresh DB starts with ZERO cases ([]), NOT Robert Chen.
2. User Registration (first_name, last_name, email, password).
3. Duplicate registration error handling.
4. User Login with valid/invalid passwords & JWT generation.
5. Current user endpoint GET /api/v1/auth/me.
6. Admin case creation POST /api/v1/admin/cases.
7. Learner case listing & detail (ensures no hidden rubric is leaked).
8. Simulation session creation.
9. Message dialogue turn with question-specific answers and DB persistence.
10. Strict User isolation (User B cannot access User A's session or messages).
11. Health (/health) and Readiness (/ready) endpoints.
"""

import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from database import Base, get_db
from models.case_model import Case, PatientProfile, ClinicalFact

# Use an isolated in-memory or file database for deterministic test execution
TEST_DB_URL = "sqlite:///./test_interactmd.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # Fresh database: Zero cases, Zero users
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_interactmd.db"):
        try:
            os.remove("./test_interactmd.db")
        except Exception:
            pass

client = TestClient(app)


def test_01_health_and_ready_endpoints():
    """Verify liveness and readiness probes."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"
    assert res_ready.json()["database"] == "connected"


def test_02_empty_database_starts_with_zero_cases():
    """CRITICAL: Fresh database MUST return [] and NEVER Robert Chen."""
    res = client.get("/api/v1/cases")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 0, f"Expected 0 cases in fresh DB, got {len(data)}"


def test_03_user_registration():
    """Test user registration with first_name, last_name, email, password."""
    payload = {
        "first_name": "Simran",
        "last_name": "Kaur",
        "email": "simran@interactmd.com",
        "password": "Password123!",
        "role": "LEARNER"
    }
    res = client.post("/api/v1/auth/register", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert data["user"]["email"] == "simran@interactmd.com"
    assert data["user"]["first_name"] == "Simran"
    assert data["user"]["last_name"] == "Kaur"
    assert data["user"]["role"] == "LEARNER"
    assert "password_hash" not in data["user"]


def test_04_duplicate_registration_rejected():
    """Test duplicate registration returns 400 Bad Request."""
    payload = {
        "first_name": "Simran",
        "last_name": "Kaur",
        "email": "simran@interactmd.com",
        "password": "AnotherPassword123!"
    }
    res = client.post("/api/v1/auth/register", json=payload)
    assert res.status_code == 400
    assert "already exists" in res.json()["detail"].lower()


def test_05_user_login():
    """Test login with valid and invalid credentials."""
    # Invalid password
    bad_login = client.post("/api/v1/auth/login", json={
        "email": "simran@interactmd.com",
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401

    # Valid password
    good_login = client.post("/api/v1/auth/login", json={
        "email": "simran@interactmd.com",
        "password": "Password123!"
    })
    assert good_login.status_code == 200
    data = good_login.json()
    assert "access_token" in data
    assert data["user"]["email"] == "simran@interactmd.com"


def test_06_current_user_profile():
    """Test GET /api/v1/auth/me returns current user info."""
    login_res = client.post("/api/v1/auth/login", json={
        "email": "simran@interactmd.com",
        "password": "Password123!"
    })
    token = login_res.json()["access_token"]

    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    user_data = me_res.json()
    assert user_data["email"] == "simran@interactmd.com"
    assert user_data["first_name"] == "Simran"
    assert "password_hash" not in user_data


def test_07_admin_case_creation_and_listing():
    """Test creating a case as ADMIN and retrieving it."""
    # Register an admin user
    admin_reg = client.post("/api/v1/auth/register", json={
        "first_name": "Chief",
        "last_name": "Educator",
        "email": "educator@interactmd.com",
        "password": "AdminPassword123!",
        "role": "ADMIN"
    })
    admin_token = admin_reg.json()["access_token"]

    # Create new case
    case_payload = {
        "title": "Severe Epigastric Pain with Radiation to Back",
        "specialty": "Gastroenterology",
        "description": "54yo male presenting with sudden onset acute epigastric pain.",
        "difficulty": "Intermediate",
        "is_published": True,
        "patient_name": "Arthur Pendelton",
        "patient_age": 54,
        "patient_gender": "Male",
        "chief_complaint": "Severe boring epigastric pain that began after dinner.",
        "history": "Pain started 2 hours ago. Radiates directly through to the back. Associated with nausea and persistent vomiting.",
        "onset": "about 2 hours ago after eating dinner.",
        "timing": "constant and unremitting.",
        "location": "in the upper central belly, going straight through to my back.",
        "character": "deep, stabbing, boring pressure.",
        "severity": "9 out of 10",
        "radiation": "straight through to my mid-back.",
        "aggravating_factors": "lying flat on my back",
        "relieving_factors": "leaning forward slightly",
        "associated_symptoms": ["Nausea", "Vomiting", "Sweating"],
        "past_medical_history": ["Gallstones diagnosed 2 years ago", "Hypertension"],
        "medications": ["Lisinopril 10mg daily"],
        "allergies": ["Penicillin - gives me a rash"],
        "diagnosis": "Acute Gallstone Pancreatitis",
        "differential_diagnosis": ["Peptic Ulcer Perforation", "Acute Cholecystitis", "Aortic Dissection"],
        "management": ["Aggressive IV fluid resuscitation", "NPO status", "IV analgesia (Fentanyl/Morphine)", "Serum Lipase STAT", "Abdominal Ultrasound"],
        "learning_objectives": ["Recognize clinical presentation of acute pancreatitis", "Order STAT Lipase and Ultrasound", "Initiate early fluid resuscitation"],
        "scoring_rubric": {"lipase_ordered": 25, "fluids_initiated": 25, "opqrst_gathered": 25, "empathy": 25}
    }

    create_res = client.post(
        "/api/v1/admin/cases",
        json=case_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert create_res.status_code == 201
    created_case = create_res.json()
    case_id = created_case["id"]
    assert created_case["title"] == "Severe Epigastric Pain with Radiation to Back"
    assert created_case["patient_name"] == "Arthur Pendelton"

    # Verify case listing returns the new case
    list_res = client.get("/api/v1/cases")
    assert list_res.status_code == 200
    cases_list = list_res.json()
    assert len(cases_list) == 1
    assert cases_list[0]["id"] == case_id
    assert cases_list[0]["patient_name"] == "Arthur Pendelton"

    # Verify case detail endpoint NEVER leaks ground truth diagnosis or rubric to learner
    detail_res = client.get(f"/api/v1/cases/{case_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "diagnosis" not in detail
    assert "scoring_rubric" not in detail
    assert "hidden_evaluation" not in detail
    assert detail["patient"]["name"] == "Arthur Pendelton"


def test_08_simulation_session_and_question_specific_ai_dialogue():
    """
    Test simulation session and verifies AI answers SPECIFIC questions without dumping entire case.
    """
    # 1. Login learner
    login_res = client.post("/api/v1/auth/login", json={
        "email": "simran@interactmd.com",
        "password": "Password123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get case ID
    cases_res = client.get("/api/v1/cases")
    case_id = cases_res.json()[0]["id"]

    # 3. Create simulation session
    sess_res = client.post("/api/v1/sessions", json={"case_id": case_id}, headers=headers)
    assert sess_res.status_code == 201
    session_id = sess_res.json()["id"]

    # 4. Ask Q1: Onset & Timing
    msg1_res = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "When did the pain start?"},
        headers=headers
    )
    assert msg1_res.status_code == 200
    reply1 = msg1_res.json()["message"]
    # Should answer onset specifically
    assert "2 hours ago" in reply1.lower() or "dinner" in reply1.lower() or "started" in reply1.lower()
    # Should NOT dump medications or allergies in onset reply
    assert "lisinopril" not in reply1.lower()
    assert "penicillin" not in reply1.lower()

    # 5. Ask Q2: Radiation
    msg2_res = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Does the pain radiate or travel anywhere?"},
        headers=headers
    )
    assert msg2_res.status_code == 200
    reply2 = msg2_res.json()["message"]
    assert "back" in reply2.lower()

    # 6. Ask Q3: Allergies
    msg3_res = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Do you have any drug allergies?"},
        headers=headers
    )
    assert msg3_res.status_code == 200
    reply3 = msg3_res.json()["message"]
    assert "penicillin" in reply3.lower() or "rash" in reply3.lower()

    # 7. Verify messages are persisted in PostgreSQL
    history_res = client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
    assert history_res.status_code == 200
    messages = history_res.json()
    # 3 learner questions + 3 patient replies = 6 messages
    assert len(messages) == 6
    assert messages[0]["sender"] == "LEARNER"
    assert messages[1]["sender"] == "PATIENT"


def test_09_strict_user_isolation():
    """
    Verify User B (Bob) CANNOT access User A (Simran)'s sessions or messages.
    """
    # Register User B
    bob_reg = client.post("/api/v1/auth/register", json={
        "first_name": "Bob",
        "last_name": "Ross",
        "email": "bob@interactmd.com",
        "password": "BobPassword123!",
        "role": "LEARNER"
    })
    bob_token = bob_reg.json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # Get Simran's session ID from previous test
    login_simran = client.post("/api/v1/auth/login", json={
        "email": "simran@interactmd.com",
        "password": "Password123!"
    })
    simran_token = login_simran.json()["access_token"]
    simran_sessions = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {simran_token}"}).json()
    assert len(simran_sessions) >= 1
    simran_session_id = simran_sessions[0]["id"]

    # Bob lists his sessions -> must be empty
    bob_sessions = client.get("/api/v1/sessions", headers=bob_headers).json()
    assert len(bob_sessions) == 0

    # Bob tries to access Simran's session detail -> 403 Forbidden
    bob_attempt_sess = client.get(f"/api/v1/sessions/{simran_session_id}", headers=bob_headers)
    assert bob_attempt_sess.status_code == 403

    # Bob tries to read Simran's messages -> 403 Forbidden
    bob_attempt_msgs = client.get(f"/api/v1/sessions/{simran_session_id}/messages", headers=bob_headers)
    assert bob_attempt_msgs.status_code == 403

    # Bob tries to send a message to Simran's session -> 403 Forbidden
    bob_attempt_send = client.post(
        f"/api/v1/sessions/{simran_session_id}/messages",
        json={"message": "Hacking into your session"},
        headers=bob_headers
    )
    assert bob_attempt_send.status_code == 403
