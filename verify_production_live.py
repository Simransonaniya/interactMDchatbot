import httpx
import uuid
import time

base_url = 'https://interactmdchatbot-1.onrender.com'
session_id = f'e2e_prod_verification_{uuid.uuid4()}'
case_id = 'chest_pain_001'

print('=====================================================')
print('RUNNING LIVE PRODUCTION FINAL CONVERSATION VERIFICATION')
print(f'Target: {base_url}')
print(f'Session ID: {session_id}')
print('=====================================================')

def send_turn(label, msg):
    resp = httpx.post(f'{base_url}/api/simulation/chat', json={
        'case_id': case_id,
        'session_id': session_id,
        'message': msg,
        'conversation_history': []
    }, timeout=45.0)
    if resp.status_code != 200:
        print(f"Error on {label}: status={resp.status_code}, body={resp.text}")
        raise ValueError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    reply = data.get('reply', '')
    print(f"\n{label}")
    print(f"Doctor: {msg}")
    print(f"Patient: {reply}")
    return data

# -------------------------------------------------------------
# EXACT 8-STEP SEQUENCE REQUIRED BY SPECIFICATION:
# Step 1: Start Robert Chen -> Initial complaint present in session
# Step 2: "When did this start and how long has it lasted?" -> ~45 minutes
# Step 3: "Had you breakfast?" -> breakfast response
# Step 4: "What had you in dinner?" -> dinner response
# Step 5: "How can you don't know?" -> challenge/clarification defense
# Step 6: "Are you sure?" -> confirmation
# Step 7: "What did you eat for lunch?" -> lunch response
# Step 8: "Why can't you remember?" -> challenge/clarification defense
# -------------------------------------------------------------

# Step 2: Timing / Onset
s2 = send_turn("Step 2 (Timing / Onset)", "When did this start and how long has it lasted?")
rep2 = s2.get('reply', '').lower()
assert "45 minutes" in rep2
assert "haven't really noticed anything like that" not in rep2

# Step 3: Breakfast
s3 = send_turn("Step 3 (Breakfast)", "Had you breakfast?")
rep3 = s3.get('reply', '').lower()
assert any(k in rep3 for k in ["breakfast", "rushing", "remember"])
assert "haven't really noticed anything like that" not in rep3

# Step 4: Dinner
s4 = send_turn("Step 4 (Dinner)", "What had you in dinner?")
rep4 = s4.get('reply', '').lower()
assert any(k in rep4 for k in ["dinner", "yesterday", "chest pain", "remember"])
assert "haven't really noticed anything like that" not in rep4

# Step 5: Ungrammatical Challenge: "How can you don't know?"
s5 = send_turn("Step 5 (Ungrammatical Challenge - How can you don't know?)", "How can you don't know?")
rep5 = s5.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep5
assert any(k in rep5 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])

# Step 6: Confirmation: "Are you sure?"
s6 = send_turn("Step 6 (Confirmation - Are you sure?)", "Are you sure?")
rep6 = s6.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep6
assert any(k in rep6 for k in ["sure", "remember", "chest pain", "focus", "dizziness", "yes"])

# Step 7: New Question: "What did you eat for lunch?"
s7 = send_turn("Step 7 (New Question - What did you eat for lunch?)", "What did you eat for lunch?")
rep7 = s7.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep7
assert any(k in rep7 for k in ["lunch", "remember", "eat", "pain"])

# Step 8: Challenge: "Why can't you remember?"
s8 = send_turn("Step 8 (Challenge - Why can't you remember?)", "Why can't you remember?")
rep8 = s8.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep8
assert any(k in rep8 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])

# -------------------------------------------------------------
# ADDITIONAL PRESERVED BEHAVIORS VALIDATION:
# -------------------------------------------------------------

# Step 9: Ungrammatical colloquial challenge ("are you made you don't know anything")
s9 = send_turn("Step 9 (Colloquial Challenge - are you made you don't know anything)", "are you made you don't know anything")
rep9 = s9.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep9
assert any(k in rep9 for k in ["rushing", "chest pain", "dizziness", "remember", "office", "overwhelmed", "pain"])

# Step 10: Current Medications History
s10 = send_turn("Step 10 (Current Medications History)", "Had you eat any medicine?")
rep10 = s10.get('reply', '').lower()
assert any(k in rep10 for k in ["amlodipine", "atorvastatin", "medications"])
assert "haven't really noticed" not in rep10

# Step 10b: Current Medications History with Typo ("had you eat any type of medician?")
s10b = send_turn("Step 10b (Medication History with Typo - had you eat any type of medician?)", "had you eat any type of medician?")
rep10b = s10b.get('reply', '').lower()
assert any(k in rep10b for k in ["amlodipine", "atorvastatin", "medications"])
assert "haven't really noticed" not in rep10b
assert "breakfast" not in rep10b and "dinner" not in rep10b

# Step 11: Management Statement
s11 = send_turn("Step 11 (Management Statement - You can take a medicine and rest)", "You can take a medicine and rest.")
rep11 = s11.get('reply', '').lower()
assert "haven't really noticed" not in rep11
assert any(k in rep11 for k in ["okay", "doctor", "rest", "relieve", "chest pain", "breathe", "help"])

# Step 12: Medication Fragment ("niciplus")
s12 = send_turn("Step 12 (Medication Fragment - niciplus)", "niciplus")
rep12 = s12.get('reply', '').lower()
assert "haven't really noticed anything like that" not in rep12
assert any(k in rep12 for k in ["medication", "medicine", "take", "doctor", "relieve", "chest pain", "breathe", "help", "okay"])

# Step 13: Demographics - PCOD in male patient
s13 = send_turn("Step 13 (Demographics - PCOD in male)", "Do you have PCOD?")
rep13 = s13.get('reply', '').lower()
assert "male" in rep13 or "doesn't apply" in rep13

# Step 14: Severity Inquiry + Confirmation
s14a = send_turn("Step 14a (Severity Inquiry)", "How severe is your discomfort?")
rep14a = s14a.get('reply', '').lower()
assert "8" in rep14a or "eight" in rep14a

s14b = send_turn("Step 14b (Severity Confirmation - Are you sure?)", "Are you sure?")
rep14b = s14b.get('reply', '').lower()
assert "8" in rep14b or "eight" in rep14b
assert "haven't really noticed anything like that" not in rep14b

print('\n=====================================================')
print('ALL 14 PRODUCTION VERIFICATION TURNS PASSED SUCCESSFULLY!')
print('=====================================================')

