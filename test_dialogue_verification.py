import asyncio
import json
from ai_orchestrator import ai_orchestrator
from mongo_db import mongo_manager

def test_all():
    case_id = "dyspnea_002"
    
    tests = [
        ("Test A: Onset", "When did your breathing difficulty start?"),
        ("Test B: Sudden vs Gradual", "Did it start suddenly or gradually?"),
        ("Test C: Medical Conditions", "Does you have any medical condition?"),
        ("Test D: Inhaler", "Have you used your inhaler today?"),
        ("Test E: Last Weekend", "What did you do last weekend?"),
        ("Test F: Favorite Food", "What is your favorite food?"),
        ("Test G1: Rapid 1", "What medical conditions do you have?"),
        ("Test G2: Rapid 2", "Have you used your inhaler today?"),
        ("Test H: Duplicate Question", "When did your breathing difficulty start?"),
    ]
    
    print("=" * 60)
    print("RUNNING INTERACTMD CLINICAL DIALOGUE VERIFICATION")
    print("=" * 60)
    
    for name, q in tests:
        res = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message=q
        )
        reply = res.get("reply", "")
        source = res.get("response_source", "")
        fact_key = res.get("fact_key", "")
        print(f"\n[{name}]")
        print(f"Doctor : {q}")
        print(f"Patient: {reply}")
        print(f"Source : {source} (Key: {fact_key})")
        
        # Grounding validations
        reply_lower = reply.lower()
        if "Test A" in name:
            assert "3 hours" in reply_lower, f"Failed onset timing test: {reply}"
            assert "panic" not in reply_lower and "anxiety" not in reply_lower, f"Hallucinated anxiety: {reply}"
        elif "Test B" in name:
            assert "3 hours" in reply_lower or "worse" in reply_lower, f"Failed sudden vs gradual test: {reply}"
            assert "panic" not in reply_lower and "anxiety" not in reply_lower, f"Hallucinated anxiety: {reply}"
        elif "Test C" in name or "Test G1" in name:
            assert "asthma" in reply_lower, f"Failed PMH test: {reply}"
            assert "check-up" not in reply_lower and "routine" not in reply_lower, f"Hallucinated routine check-up: {reply}"
        elif "Test D" in name or "Test G2" in name:
            assert "inhaler" in reply_lower and ("four times" in reply_lower or "4 times" in reply_lower or "used" in reply_lower), f"Failed inhaler test: {reply}"
            assert "remembering" not in reply_lower and "correctly" not in reply_lower, f"Hallucinated compliance: {reply}"
        elif "Test E" in name:
            assert "friends" not in reply_lower and "movie" not in reply_lower and "dinner" not in reply_lower, f"Hallucinated weekend: {reply}"
            assert source == "UNKNOWN", f"Expected source UNKNOWN for unmentioned past activities, got {source}"
        elif "Test F" in name:
            assert "relate" in reply_lower or "experiencing" in reply_lower, f"Failed out of scope test: {reply}"
            assert source == "UNKNOWN", f"Expected source UNKNOWN for favorite food, got {source}"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_all()
