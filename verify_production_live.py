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
    }, timeout=30.0)
    if resp.status_code != 200:
        print(f"Error on {label}: status={resp.status_code}, body={resp.text}")
        raise ValueError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    print(f"{label}: {data.get('reply')}")
    return data

# Turn 1: Start Case / Opening
t1 = send_turn("Turn 1 (Opening)", "Hello, what brought you to the emergency room today?")
assert 'pressure' in t1.get('reply', '').lower() or 'chest' in t1.get('reply', '').lower()

# Turn 2: Severity
t2 = send_turn("Turn 2 (Severity)", "On a scale of 1 to 10, how severe is your discomfort right now?")
assert '8' in t2.get('reply', '') or 'eight' in t2.get('reply', '').lower()
assert "haven't really noticed" not in t2.get('reply', '').lower()

# Turn 3: Repeated Severity
t3 = send_turn("Turn 3 (Repeated Severity)", "On a scale of 1 to 10, how severe is your discomfort right now?")
assert '8' in t3.get('reply', '') or 'eight' in t3.get('reply', '').lower()
assert "haven't really noticed" not in t3.get('reply', '').lower()

# Turn 4: "are you sure?"
t4 = send_turn("Turn 4 (Clarification 'are you sure?')", "are you sure?")
reply4 = t4.get('reply', '').lower()
assert "haven't really noticed" not in reply4
assert "anything like that" not in reply4
assert any(k in reply4 for k in ["yes", "sure", "8", "intense", "severe"])

# Turn 5: Management: "you should take rest"
t5 = send_turn("Turn 5 (Management Statement 'you should take rest')", "you should take rest")
reply5 = t5.get('reply', '').lower()
assert "haven't really noticed" not in reply5
assert "anything like that" not in reply5
assert any(k in reply5 for k in ["rest", "sit down", "okay", "ease", "help", "chest"])

# Turn 6: Treatment: "you should take tablet"
t6 = send_turn("Turn 6 (Medication Statement 'you should take tablet')", "you should take tablet")
reply6 = t6.get('reply', '').lower()
assert "haven't really noticed" not in reply6
assert "anything like that" not in reply6
assert any(k in reply6 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "tablet", "medicine"])

# Turn 7: Medication with typo: "take paracetomol"
t7 = send_turn("Turn 7 (Medication Statement 'take paracetomol')", "take paracetomol")
reply7 = t7.get('reply', '').lower()
assert "haven't really noticed" not in reply7
assert "anything like that" not in reply7
assert any(k in reply7 for k in ["okay", "doctor", "relieve", "chest pain", "breathe", "paracetamol", "paracetomol"])

# Turn 8: New History Question: Radiation
t8 = send_turn("Turn 8 (New History Question - Radiation)", "Does the pain spread anywhere?")
reply8 = t8.get('reply', '').lower()
assert any(k in reply8 for k in ['jaw', 'arm', 'left', 'radiat'])

# Turn 9: Repeated History Question: Radiation
t9 = send_turn("Turn 9 (Repeated History Question - Radiation)", "Does the pain spread to your jaw or arm?")
reply9 = t9.get('reply', '').lower()
assert any(k in reply9 for k in ['jaw', 'arm', 'yes', 'spread', 'radiat'])

# Turn 10: Examination
t10 = httpx.post(f'{base_url}/api/simulation/exam', json={
    'case_id': case_id,
    'session_id': session_id,
    'exam_id': 'pf-cardiovascular-01'
}, timeout=30.0).json()
print(f"Turn 10 (Physical Exam): {t10.get('finding', t10.get('value', 'OK'))}")
assert t10.get('finding') or t10.get('value') or t10.get('exam_id')

print('=====================================================')
print('ALL 10 PRODUCTION SEQUENCE STEPS VERIFIED & PASSED!')
print('=====================================================')
