# InteractMD — Python AI Patient Chatbot Backend

यह InteractMD का **Dedicated Python Backend** है, जो AI Virtual Patient Chatbot और Clinical Evaluation Engine को चलाता है।

---

## 📁 Folder Structure (फाइलें कहाँ क्या हैं)

```
chatbot_backend/
├── main.py            # FastAPI Application (API Endpoints & Routing)
├── ai_engine.py       # AI Patient Dialogue Engine (OPQRST, Empathy, Layperson Persona)
├── cases_data.py      # Clinical Benchmark Cases (Ground Truth Data)
├── evaluator.py       # 5-Dimension Attending Physician Evaluation Engine
├── schemas.py         # Pydantic Request & Response Data Models
├── run.py             # Server Runner Script
├── requirements.txt   # Required Python Packages
└── README.md          # Guide & Documentation
```

---

## 🚀 How to Run the Backend (कैसे चलाएं)

### Command 1: Runner Script (सबसे आसान)
```bash
cd chatbot_backend
python run.py
```

### Command 2: Uvicorn Direct
```bash
cd chatbot_backend
uvicorn main:app --reload --port 8000
```

Backend **`http://localhost:8000`** पर शुरू हो जाएगा।

---

## 🌐 Endpoints & API Documentation

- **Swagger UI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: `GET http://localhost:8000/api/health`
- **Patient Chat**: `POST http://localhost:8000/api/simulation/chat`
- **Submit Evaluation**: `POST http://localhost:8000/api/simulation/evaluate`
- **Physical Exam**: `POST http://localhost:8000/api/simulation/exam`
- **Investigations**: `POST http://localhost:8000/api/simulation/investigation`

---

## 🔗 Frontend Integration (Frontend से कैसे जुड़ा है)

आपका React / Vite Frontend (`src/services/apiClient.ts`) सीधे `http://localhost:8000/api` पर कॉल करता है:
1. जब Python backend चालू होता है, तो Frontend के Simulation Room में **"AI Backend Online"** का Green badge दिखने लगता है।
2. जब आप AI Patient से चैट करते हैं, तो सवाल `POST /api/simulation/chat` पर Python backend को भेजा जाता है, और Python AI Patient यथार्थवादी मरीज़ की तरह जवाब देता है।
3. जब आप "Submit Diagnosis & Evaluate" पर क्लिक करते हैं, तो `POST /api/simulation/evaluate` पर Attending Physician 5 OSCE Competency Dimensions (Interview, Reasoning, Empathy, Safety, Management) पर स्कोर कार्ड तैयार करता है।
