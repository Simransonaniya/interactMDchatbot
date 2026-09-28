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

# Turn 1: Shortness of breath
t1 = send_turn("Turn 1 (Dyspnea / Wheezing)", "Are you feeling short of breath or wheezing?")
reply1 = t1.get('reply', '').lower()
assert any(k in reply1 for k in ['short of breath', 'breath', 'tight'])
assert "haven't really noticed" not in reply1

# Turn 2: Breakfast
t2 = send_turn("Turn 2 (Breakfast Query)", "Had you breakfast?")
reply2 = t2.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply2
assert any(k in reply2 for k in ['breakfast', 'morning', 'remember', 'ate', 'rushing'])

# Turn 3: Temporal / Meal mismatch (Lunch & Dinner yesterday - must NOT return breakfast!)
t3 = send_turn("Turn 3 (Lunch & Dinner Yesterday Query)", "What did you eat yesterday for lunch and dinner?")
reply3 = t3.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply3
assert "breakfast" not in reply3
assert any(k in reply3 for k in ['lunch', 'dinner'])
assert 'yesterday' in reply3

# Turn 4: Current Medications History
t4 = send_turn("Turn 4 (Current Medications History)", "What medicines do you take?")
reply4 = t4.get('reply', '').lower()
assert any(k in reply4 for k in ['amlodipine', 'atorvastatin', 'medications'])
assert "haven't really noticed" not in reply4

# Turn 5: Management Statement ("You can take a medicine and rest.")
t5 = send_turn("Turn 5 (Management Statement)", "You can take a medicine and rest.")
reply5 = t5.get('reply', '').lower()
assert "haven't really noticed" not in reply5
assert any(k in reply5 for k in ['okay', 'doctor', 'rest', 'relieve', 'chest pain', 'breathe', 'help'])

# Turn 6: Medication Fragment ("Niciplus.")
t6 = send_turn("Turn 6 (Medication Fragment - Niciplus)", "Niciplus.")
reply6 = t6.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply6
assert any(k in reply6 for k in ['medication', 'medicine', 'take', 'doctor', 'relieve', 'chest pain', 'breathe', 'help', 'okay'])

# Turn 7: Medication Name + Form with typo ("Paracetomol tablet.")
t7 = send_turn("Turn 7 (Medication Fragment with typo - Paracetomol tablet)", "Paracetomol tablet.")
reply7 = t7.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply7
assert any(k in reply7 for k in ['okay', 'doctor', 'relieve', 'chest pain', 'breathe', 'tablet', 'medicine', 'take'])

# Turn 8: PCOD in male patient ("Do you have PCOD?")
t8 = send_turn("Turn 8 (Demographics - PCOD in male)", "Do you have PCOD?")
reply8 = t8.get('reply', '').lower()
assert "male" in reply8 or "doesn't apply" in reply8

# Turn 9: Clarification ("Are you sure?")
t9 = send_turn("Turn 9 (Clarification - Are you sure?)", "Are you sure?")
reply9 = t9.get('reply', '').lower()
assert "haven't really noticed" not in reply9
assert any(k in reply9 for k in ['yes', 'sure', 'doctor', 'definitely', 'honest'])

# Turn 10a: Repeated history question (Severity 1)
t10a = send_turn("Turn 10a (Severity Inquiry 1)", "How severe is your discomfort?")
reply10a = t10a.get('reply', '').lower()
assert '8' in reply10a or 'eight' in reply10a

# Turn 10b: Repeated history question (Severity 2)
t10b = send_turn("Turn 10b (Severity Inquiry 2 - Repeated)", "How severe is your discomfort?")
reply10b = t10b.get('reply', '').lower()
assert '8' in reply10b or 'eight' in reply10b

print('\n=====================================================')
print('ALL 10 PRODUCTION TURNS VERIFIED AND PASSED!')
print('=====================================================')
