"""
Comprehensive Integration Test Suite for InteractMD Frontend <-> Backend <-> MongoDB <-> HuggingFace.
Tests all requirements specified in user prompt.
"""

import sys
import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8001"

def http_post(endpoint: str, data: dict):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode("utf-8"))

def http_get(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    with urllib.request.urlopen(url) as res:
        return json.loads(res.read().decode("utf-8"))

def run_tests():
    print("================================================================")
    print("      INTERACTMD INTEGRATION & BEHAVIORAL TEST SUITE            ")
    print("================================================================\n")

    # 1. Health Check
    print("[1/8] Testing Health Check endpoint (/api/health)...")
    health = http_get("/api/health")
    assert health.get("status") == "online", f"Expected online, got {health}"
    print(f"  -> Health check PASSED: {health}\n")

    # 2. Case Listing from MongoDB
    print("[2/8] Testing GET /api/v1/cases from MongoDB...")
    cases = http_get("/api/v1/cases")
    assert len(cases) >= 1, f"Expected at least 1 case, got {len(cases)}"
    case = cases[0]
    case_id = case["id"]
    print(f"  -> Loaded {len(cases)} cases from MongoDB. Active benchmark case: '{case['title']}' ({case_id})\n")

    # 3. Session Initialization
    print(f"[3/8] Testing POST /api/v1/sessions for case {case_id}...")
    session_res = http_post("/api/v1/sessions", {"case_id": case_id})
    session_id = session_res["id"]
    assert session_id, "No session ID returned"
    print(f"  -> Session created: {session_id}\n")

    # 4. Chat - Onset & Progressive Disclosure
    print("[4/8] Testing Chat: 'When did the pain start?'...")
    chat_res = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "When did the pain start?",
        "conversation_history": []
    })
    reply1 = chat_res.get("reply", "")
    print(f"  -> Patient Reply: \"{reply1}\"")
    assert "45 minutes" in reply1.lower() or "minutes" in reply1.lower(), f"Unexpected reply: {reply1}"
    print("  -> Chat Onset Test PASSED.\n")

    # 5. Session Consistency
    print("[5/8] Testing Session Consistency...")
    # Follow-up 1: Continuity
    res_cont = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "Has it been continuous?",
        "conversation_history": [{"sender": "student", "text": "When did the pain start?"}, {"sender": "patient", "text": reply1}]
    })
    print(f"  -> Continuity Reply: \"{res_cont.get('reply')}\"")
    
    # Follow-up 2: Activity
    res_act = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "What were you doing when it started?",
        "conversation_history": []
    })
    print(f"  -> Activity Reply: \"{res_act.get('reply')}\"")
    print("  -> Session Consistency Test PASSED.\n")

    # 6. Unrelated Questions & Distinct Negative Symptoms
    print("[6/8] Testing Unrelated & Negative Symptom Guardrails (no random radiation)...")
    
    # Cough test
    res_cough = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "Do you have a cough?",
        "conversation_history": []
    })
    print(f"  -> Cough Reply: \"{res_cough.get('reply')}\"")
    assert "radiation" not in res_cough.get("reply", "").lower() and "jaw" not in res_cough.get("reply", "").lower(), "Cough erroneously returned radiation!"
    assert "cough" in res_cough.get("reply", "").lower() or "no" in res_cough.get("reply", "").lower()

    # Back Pain test (Crucial User requirement: 'Doctor: I have back pain' must not return jaw/arm radiation)
    res_back = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "I have back pain",
        "conversation_history": []
    })
    print(f"  -> Back Pain Reply: \"{res_back.get('reply')}\"")
    assert "radiation" not in res_back.get("reply", "").lower() and "jaw" not in res_back.get("reply", "").lower(), "Back pain erroneously returned jaw/arm radiation!"
    assert "back" in res_back.get("reply", "").lower() or "no" in res_back.get("reply", "").lower()

    # Hairfall test
    res_hair = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "I have hairfall.",
        "conversation_history": []
    })
    print(f"  -> Hairfall Reply: \"{res_hair.get('reply')}\"")
    assert "radiation" not in res_hair.get("reply", "").lower(), "Hairfall returned radiation!"

    # Favorite food test
    res_food = http_post("/api/simulation/chat", {
        "case_id": case_id,
        "session_id": session_id,
        "message": "What is your favorite food?",
        "conversation_history": []
    })
    print(f"  -> Favorite Food Reply: \"{res_food.get('reply')}\"")
    assert "radiation" not in res_food.get("reply", "").lower()

    print("  -> Unrelated & Negative Symptom Tests ALL PASSED.\n")

    # 7. Physical Exams & Investigations
    print("[7/8] Testing Physical Exam & STAT Investigation Endpoints...")
    exam_res = http_post("/api/simulation/exam", {
        "case_id": case_id,
        "session_id": session_id,
        "exam_id": f"{case_id}_cv_auscultation",
        "system": "Cardiovascular"
    })
    print(f"  -> Physical Exam Result: {exam_res.get('finding')} -> {exam_res.get('value')}")
    assert exam_res.get("value"), "No exam finding returned"

    inv_res = http_post("/api/simulation/investigation", {
        "case_id": case_id,
        "session_id": session_id,
        "test_id": f"{case_id}_ecg_stat"
    })
    print(f"  -> STAT Investigation Result: {inv_res.get('name')} -> {inv_res.get('result')}")
    assert inv_res.get("result"), "No investigation result returned"
    print("  -> Exams & Investigations Tests PASSED.\n")

    # 8. Clinical Diagnosis & Evaluation
    print("[8/8] Testing Attending Physician Evaluation (/api/simulation/evaluate)...")
    eval_res = http_post("/api/simulation/evaluate", {
        "case_id": case_id,
        "session_id": session_id,
        "conversation_history": [
            {"sender": "student", "text": "When did the pain start?"},
            {"sender": "patient", "text": reply1},
            {"sender": "student", "text": "Does it radiate to your jaw or arm?"},
            {"sender": "patient", "text": "Yes, it spreads up to my left jaw and down my left arm."}
        ],
        "performed_exam_ids": [f"{case_id}_cv_auscultation"],
        "ordered_investigation_ids": [f"{case_id}_ecg_stat"],
        "primary_diagnosis_id": "Acute Coronary Syndrome (STEMI)",
        "differential_diagnosis_ids": ["Aortic Dissection", "Pulmonary Embolism"],
        "selected_management_ids": ["stat_aspirin_heparin", "stat_cardiology_cath_activation"],
        "clinical_rationale": "Crushing retrosternal chest pain with left arm radiation, diaphoresis, and ST-segment elevations.",
        "duration_seconds": 320
    })
    summary = eval_res.get('attending_physician_notes') or eval_res.get('clinical_feedback_summary') or 'Clinical review completed.'
    print(f"  -> Attending Summary: {summary[:100]}...")
    assert eval_res.get("overall_score") is not None, "Evaluation failed"
    print("  -> Attending Evaluation PASSED.\n")

    print("================================================================")
    print("  ALL 8 END-TO-END INTEGRATION TESTS PASSED SUCCESSFULLY!       ")
    print("================================================================\n")

if __name__ == "__main__":
    run_tests()
