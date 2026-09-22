"""
InteractMD — Python AI Patient Chatbot Backend Runner.
Run directly with: python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("\n========================================================")
    print("  INTERACTMD PYTHON AI PATIENT CHATBOT BACKEND")
    print("  Server starting on: http://localhost:8000")
    print("  Swagger Documentation: http://localhost:8000/docs")
    print("========================================================\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
