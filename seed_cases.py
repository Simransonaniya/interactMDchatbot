"""
InteractMD — MongoDB Clinical Case Seeder.
Explicitly populates MongoDB Atlas from modular JSON specifications in chatbot/data/cases.json.
Does NOT automatically overwrite unless --force is specified.
"""

import os
import sys
import json
import time
import argparse
from mongo_db import mongo_manager


DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "cases.json")

def load_benchmark_cases():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load {DATA_FILE}: {e}")
    return []

BENCHMARK_CASES = load_benchmark_cases()


def seed_database(force: bool = False):
    print("========================================================")
    print("  InteractMD — MongoDB Clinical Case Seeder")
    print("========================================================")
    
    if not mongo_manager.is_connected:
        print("[Error] Could not connect to MongoDB Atlas. Check your MONGODB_URI.")
        sys.exit(1)

    existing_count = mongo_manager.cases_col.count_documents({})
    print(f"Current cases in MongoDB: {existing_count}")

    cases = load_benchmark_cases()
    if not cases:
        print(f"[Error] No cases found in {DATA_FILE}")
        return

    if existing_count > 0 and not force:
        print(f"Database already contains {existing_count} case(s).")
        print("To overwrite/re-seed, run: python seed_cases.py --force")
        return

    print(f"Seeding {len(cases)} authentic clinical cases from JSON into MongoDB...")
    for case in cases:
        saved_id = mongo_manager.save_case(case)
        print(f"  [+] Saved case: '{case.get('case_id') or case.get('id')}' - {case.get('title')} (Patient: {case.get('patient', {}).get('name')})")

    total = mongo_manager.cases_col.count_documents({})
    print("========================================================")
    print(f"  [Success] Seeded MongoDB Atlas successfully from JSON! Total cases: {total}")
    print("========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed InteractMD clinical cases from JSON into MongoDB")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing clinical cases")
    args = parser.parse_args()
    seed_database(force=args.force)
