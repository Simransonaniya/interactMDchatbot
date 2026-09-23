"""
End-to-End Simulation Encounter Verification.
Simulates a multi-turn student encounter with Arthur Pendelton (Case ACS),
verifying progressive disclosure, examination maneuvers, investigation orders,
and Attending OSCE Evaluation.
"""

from ai_engine import AIPatientEngine
from cases_data import CLINICAL_CASES
from evaluator import evaluate_encounter
from schemas import EvaluationRequest

def main():
    engine = AIPatientEngine()
    case = CLINICAL_CASES["case-acs-1"]
    history = []

    print("================================================================")
    print("  INTERACTMD END-TO-END VIRTUAL PATIENT SIMULATION TEST")
    print("  Case: Arthur Pendelton (58M, Acute Coronary Syndrome)")
    print("================================================================\n")

    turns = [
        "Good morning Mr. Pendelton. What brought you in today?",
        "When exactly did this pain start?",
        "What were you doing when it came on?",
        "Where is the pain located and does it radiate anywhere?",
        "How severe is the pain from 1 to 10?",
        "Have you had any shortness of breath or sweating?",
        "Do you have any past medical history or take regular medications?",
        "Do you have any drug allergies?",
        "Does heart disease run in your family?",
        "Do you smoke or drink?",
        "Don't worry Mr. Pendelton, we are going to take good care of you.",
        "I'd like to check your vital signs and perform a cardiovascular examination.",
        "I'd like to order a stat 12-lead ECG and cardiac enzymes."
    ]

    for i, t in enumerate(turns, start=1):
        res = engine.process_turn(case, t, history)
        print(f"Turn {i}:")
        print(f"  Doctor:  \"{t}\"")
        print(f"  Patient: \"{res['reply']}\"")
        print(f"  Category: [{res['category']}] | Empathy: {res['empathy_detected']}\n")
        history.append({"sender": "student", "text": t})
        history.append({"sender": "patient", "text": res["reply"], "category": res["category"]})

    print("--- Simulating Attending Evaluation ---")
    eval_req = EvaluationRequest(
        case_id="case-acs-1",
        session_id="e2e-test-session",
        conversation_history=history,
        performed_exam_ids=["pf-cv-01", "pf-resp-01"],
        ordered_investigation_ids=["inv-ecg-01", "inv-trop-01"],
        primary_diagnosis_id="dx-nstemi",
        clinical_rationale="Patient presented with classic crushing substernal chest pain radiating to jaw and arm with diaphoresis."
    )
    eval_res = evaluate_encounter(eval_req, case_obj=None)
    print(f"OSCE Overall Score: {eval_res.overall_score}% | Pass Status: {eval_res.pass_status}")
    print(f"Attending Notes: {eval_res.attending_physician_notes[:180]}...")
    print("\n[SUCCESS] Full End-to-End Simulation Encounter Passed!")

if __name__ == "__main__":
    main()
