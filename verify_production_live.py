import httpx
import uuid
import time

base_url = 'https://interactmdchatbot-1.onrender.com'
session_id = f'e2e_prod_verification_{uuid.uuid4()}'
case_id = 'chest_pain_001'

print('=====================================================')
print('RUNNING LIVE PRODUCTION END-TO-END VERIFICATION')
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

# Turn 1: Start Case / Initial greeting
t1 = send_turn("Turn 1 (Opening statement)", "Hello, what brought you into the clinic today?")
reply1 = t1.get('reply', '').lower()
assert any(k in reply1 for k in ['chest', 'elephant', 'pressure', 'tightness'])

# Turn 2: Can you describe what the pain feels like?
t2 = send_turn("Turn 2 (Pain Description)", "Can you describe what the pain feels like?")
reply2 = t2.get('reply', '').lower()
assert any(k in reply2 for k in ['elephant', 'pressure', 'heavy', 'tight', 'crushing'])
assert "haven't really noticed" not in reply2
assert "office" not in reply2 or "sitting" in reply2 # verify it's not the entire multi-sentence statement dumped verbatim

# Turn 3: Cold sweats, nausea, vomiting
t3 = send_turn("Turn 3 (Associated Symptoms)", "Have you experienced any cold sweats, nausea, or vomiting?")
reply3 = t3.get('reply', '').lower()
assert any(k in reply3 for k in ['sweat', 'cold sweat', 'dizzy'])
assert "haven't really noticed" not in reply3

# Turn 4: Management: you should take medicine home and take rest
t4 = send_turn("Turn 4 (Management Statement)", "you should take medicine home and take rest")
reply4 = t4.get('reply', '').lower()
assert "haven't really noticed" not in reply4
assert "anything like that" not in reply4
assert any(k in reply4 for k in ['okay', 'doctor', 'rest', 'relieve', 'help', 'chest'])

# Turn 5: Medication: you can take a Paracetamol if you feel like a fever
t5 = send_turn("Turn 5 (Medication Statement)", "you can take a Paracetamol if you feel like a fever")
reply5 = t5.get('reply', '').lower()
assert "haven't really noticed" not in reply5
assert "anything like that" not in reply5
assert any(k in reply5 for k in ['okay', 'doctor', 'relieve', 'chest', 'breathe', 'paracetamol', 'fever', 'medicine'])

# Turn 6: PCOD question to male patient
t6 = send_turn("Turn 6 (Demographic Applicability - PCOD)", "do you have a PCOD also")
reply6 = t6.get('reply', '').lower()
assert any(k in reply6 for k in ['male', 'man', "doesn't apply", 'not applicable', 'female'])

# Turn 7: Breakfast question (had you breakfast)
t7 = send_turn("Turn 7 (Diet/Breakfast Question 1)", "had you breakfast")
reply7 = t7.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply7
assert any(k in reply7 for k in ["breakfast", "ate", "eat", "remember", "sure", "morning", "rushing"])

# Turn 8: Breakfast detail question (what was you eat in your breakfast)
t8 = send_turn("Turn 8 (Diet/Breakfast Question 2)", "what was you eat in your breakfast")
reply8 = t8.get('reply', '').lower()
assert "haven't really noticed anything like that" not in reply8
assert any(k in reply8 for k in ["breakfast", "ate", "eat", "remember", "sure", "morning", "food"])

# Turn 9: Clarification: are you sure?
t9 = send_turn("Turn 9 (Clarification - Are you sure?)", "are you sure?")
reply9 = t9.get('reply', '').lower()
assert "haven't really noticed" not in reply9
assert any(k in reply9 for k in ["yes", "sure", "doctor", "remember", "morning", "really", "honest"])

# Turn 10a: Severity (first time)
t10a = send_turn("Turn 10a (Severity 1)", "How severe is your discomfort?")
reply10a = t10a.get('reply', '').lower()
assert '8' in reply10a or 'eight' in reply10a
assert "haven't really noticed" not in reply10a

# Turn 10b: Repeated Severity (second time)
t10b = send_turn("Turn 10b (Severity 2 - Repeated)", "How severe is your discomfort?")
reply10b = t10b.get('reply', '').lower()
assert '8' in reply10b or 'eight' in reply10b
assert "haven't really noticed" not in reply10b

print('\n=====================================================')
print('ALL 10 PRODUCTION SEQUENCE TURNS VERIFIED & PASSED!')
print('=====================================================')
