import urllib.request
import json

BASE_URL = "http://localhost:8000/api"

def main():
    print("\n--- Testing Python AI Patient Backend ---")
    
    # 1. Health
    with urllib.request.urlopen(f"{BASE_URL}/health") as r:
        health = json.loads(r.read())
        print(f"[1] Health check: {health}")
        assert health["status"] == "online"

    # 2. Chat (Onset)
    req1 = urllib.request.Request(
        f"{BASE_URL}/simulation/chat",
        data=json.dumps({
            "case_id": "case-acs-1",
            "message": "I am so sorry you are in such pain Mr. Chen. Can you describe when this started?",
            "conversation_history": []
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req1) as r:
        res1 = json.loads(r.read())
        print(f"[2] Patient Chat (Onset & Empathy):")
        print(f"    Reply: \"{res1['reply']}\"")
        print(f"    Empathy Detected: {res1['empathy_detected']}")
        print(f"    Category: {res1['category']}")
        assert res1["empathy_detected"] is True

    # 3. Chat (Radiation)
    req2 = urllib.request.Request(
        f"{BASE_URL}/simulation/chat",
        data=json.dumps({
            "case_id": "case-acs-1",
            "message": "Does the pain radiate anywhere else, like your jaw or left arm?",
            "conversation_history": []
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req2) as r:
        res2 = json.loads(r.read())
        print(f"[3] Patient Chat (Radiation):")
        print(f"    Reply: \"{res2['reply']}\"")
        print(f"    Category: {res2['category']}")

    # 4. Physical Exam
    req_exam = urllib.request.Request(
        f"{BASE_URL}/simulation/exam",
        data=json.dumps({
            "case_id": "case-acs-1",
            "system": "Cardiovascular"
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_exam) as r:
        res_exam = json.loads(r.read())
        print(f"[4] Physical Exam Findings: {len(res_exam['findings'])} items returned")

    # 5. Diagnostic Investigation
    req_inv = urllib.request.Request(
        f"{BASE_URL}/simulation/investigation",
        data=json.dumps({
            "case_id": "case-acs-1",
            "test_id": "inv-ecg"
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_inv) as r:
        res_inv = json.loads(r.read())
        print(f"[5] Investigation Result: {res_inv['test']} -> {res_inv['result']['value']}")

    # 6. Evaluation
    req_eval = urllib.request.Request(
        f"{BASE_URL}/simulation/evaluate",
        data=json.dumps({
            "case_id": "case-acs-1",
            "conversation_history": [
                {"sender": "student", "text": "I am so sorry you are in pain. When did it start?"},
                {"sender": "student", "text": "Does it radiate to your left arm or jaw?"}
            ],
            "performed_exam_ids": ["exam-cv"],
            "ordered_investigation_ids": ["inv-ecg"],
            "primary_diagnosis_id": "dx-stemi",
            "differential_diagnosis_ids": ["dx-dissection", "dx-pe"],
            "selected_management_ids": ["mgmt-aspirin", "mgmt-cath", "mgmt-heparin"],
            "clinical_rationale": "Patient presents with classic substernal crushing pain with ST-segment elevation on 12-lead ECG in inferior leads.",
            "duration_seconds": 240
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_eval) as r:
        res_eval = json.loads(r.read())
        print(f"[6] Clinical Evaluation:")
        print(f"    Overall Score: {res_eval['overall_score']}%")
        print(f"    Pass Status: {res_eval['pass_status']}")
        print(f"    Attending Summary: {res_eval['attending_physician_notes']}")

    print("\n>>> ALL PYTHON BACKEND ENDPOINTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    main()
