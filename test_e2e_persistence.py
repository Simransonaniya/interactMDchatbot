"""
End-to-end verification script for InteractMD Database Persistence.
Simulates a full student OSCE workflow and verifies that ALL data is written
into SQLite tables ('simulation_sessions', 'messages', 'interaction_events', 'evaluations', 'users', 'cases').
"""

import os
import sys
import json
import sqlite3
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))

from main import app
from database import engine, Base
from mongo_db import mongo_manager

client = TestClient(app)

def test_full_database_persistence():
    print("========================================================")
    print("  Testing End-to-End Simulation Database Persistence")
    print("========================================================")

    # 1. User Registration & Login
    print("\n1. Testing User Registration & Persistence...")
    reg_resp = client.post("/api/v1/auth/register", json={
        "first_name": "Test",
        "last_name": "Student",
        "email": "test_student@med.edu",
        "password": "securepassword123"
    })
    print(f"  Status: {reg_resp.status_code}")
    assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"
    token = reg_resp.json()["access_token"]
    user_id = reg_resp.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"  [PASS] User registered (ID: {user_id})")

    # 2. Cases Listing
    print("\n2. Testing Case Retrieval from Database...")
    cases_resp = client.get("/api/v1/cases", headers=headers)
    assert cases_resp.status_code == 200
    cases = cases_resp.json()
    assert len(cases) > 0, "No cases retrieved"
    case_id = cases[0]["id"]
    print(f"  [PASS] Loaded {len(cases)} cases. Target Case: {case_id} ({cases[0]['title']})")

    # 3. Create Simulation Session
    print("\n3. Testing Simulation Session Creation...")
    sess_resp = client.post("/api/v1/sessions", json={"case_id": case_id}, headers=headers)
    assert sess_resp.status_code == 201, f"Session create failed: {sess_resp.text}"
    session_id = sess_resp.json()["id"]
    print(f"  [PASS] Session created: {session_id}")

    # 4. Chat Dialogue Turn (Learner -> Patient)
    print("\n4. Testing Dialogue Message Persistence...")
    msg_resp = client.post(f"/api/v1/sessions/{session_id}/messages", json={
        "message": "When did the chest tightness start, and does it radiate anywhere?"
    }, headers=headers)
    assert msg_resp.status_code == 200, f"Chat message failed: {msg_resp.text}"
    patient_reply = msg_resp.json()["message"]
    print(f"  [PASS] Patient reply received: '{patient_reply[:60]}...'")

    # Second dialogue turn via /api/simulation/chat
    chat_resp2 = client.post("/api/simulation/chat", json={
        "case_id": case_id,
        "session_id": session_id,
        "message": "How severe is the pain on a scale from 1 to 10?"
    }, headers=headers)
    assert chat_resp2.status_code == 200
    print(f"  [PASS] Simulation chat reply: '{chat_resp2.json()['reply'][:60]}...'")

    # 5. Physical Examination
    print("\n5. Testing Physical Examination Persistence...")
    exam_resp = client.post(f"/api/v1/sessions/{session_id}/examinations", json={
        "case_id": case_id,
        "exam_id": "exam-cv",
        "system": "Cardiovascular"
    }, headers=headers)
    assert exam_resp.status_code == 200, f"Exam failed: {exam_resp.text}"
    print(f"  [PASS] Exam result: {exam_resp.json().get('finding') or exam_resp.json().get('name')}")

    # 6. Diagnostic Investigation
    print("\n6. Testing Diagnostic Investigation Persistence...")
    inv_resp = client.post(f"/api/v1/sessions/{session_id}/investigations", json={
        "case_id": case_id,
        "test_id": "inv-ecg"
    }, headers=headers)
    assert inv_resp.status_code == 200, f"Investigation failed: {inv_resp.text}"
    print(f"  [PASS] Investigation ordered: {inv_resp.json().get('name')}")

    # 7. Diagnosis Submission
    print("\n7. Testing Differential & Working Diagnosis Submission...")
    dx_resp = client.post(f"/api/v1/sessions/{session_id}/diagnosis", json={
        "most_likely": "Acute Inferior STEMI",
        "differential": ["Acute Inferior STEMI", "Aortic Dissection", "Pulmonary Embolism"],
        "reasoning": "Classic retrosternal crushing pain radiating to left jaw with ST-elevation."
    }, headers=headers)
    assert dx_resp.status_code == 200
    print(f"  [PASS] Diagnosis submission recorded.")

    # 8. Management Submission
    print("\n8. Testing Management Plan Submission...")
    mgmt_resp = client.post(f"/api/v1/sessions/{session_id}/management", json={
        "immediate_actions": ["Aspirin 325mg chewable", "STAT Cardiac Catheterization Activation"],
        "treatment": ["Oxygen", "Sublingual Nitroglycerin", "Heparin"],
        "escalation": "Cardiology consult"
    }, headers=headers)
    assert mgmt_resp.status_code == 200
    print(f"  [PASS] Management submission recorded.")

    # 9. Attending OSCE Evaluation
    print("\n9. Testing Attending OSCE Evaluation Persistence...")
    eval_resp = client.post(f"/api/v1/sessions/{session_id}/evaluate", json={
        "case_id": case_id,
        "session_id": session_id,
        "duration_seconds": 180
    }, headers=headers)
    assert eval_resp.status_code == 200, f"Evaluate failed: {eval_resp.text}"
    eval_data = eval_resp.json()
    score = eval_data.get("overall_score", 0)
    print(f"  [PASS] Attending evaluation complete. Overall Score: {score}/100 (Pass: {eval_data.get('pass_status')})")

    # 10. Direct SQLite Database Inspection (Verification of Table Records)
    print("\n10. Inspecting SQLite Database Table Records...")
    db_path = os.path.join(os.path.dirname(__file__), "interactmd.db")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = ["users", "cases", "simulation_sessions", "messages", "interaction_events", "evaluations"]
    counts = {}
    for t in tables:
        cnt = cur.execute(f"SELECT COUNT(*) FROM `{t}`").fetchone()[0]
        counts[t] = cnt
        print(f"  [*] Table '{t}': {cnt} row(s)")

    con.close()

    assert counts["simulation_sessions"] >= 1, "Simulation session was NOT saved in database!"
    assert counts["messages"] >= 2, "Messages were NOT saved in database!"
    assert counts["interaction_events"] >= 1, "Interaction events were NOT saved in database!"
    assert counts["evaluations"] >= 1, "Evaluation was NOT saved in database!"
    assert counts["users"] >= 3, "Users were NOT saved in database!"

    print("\n========================================================")
    print("  [ALL TESTS PASSED] All Simulation Data Persisted to Database!")
    print("========================================================")


if __name__ == "__main__":
    test_full_database_persistence()
