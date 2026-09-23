"""
InteractMD — Python AI Patient Chatbot Backend Runner.
Run directly with: python run.py
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    print("\n========================================================")
    print("  INTERACTMD PYTHON AI PATIENT CHATBOT BACKEND")
    print(f"  Server starting on: http://localhost:{port}")
    print(f"  Swagger Documentation: http://localhost:{port}/docs")
    print("========================================================\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
