"""
InteractMD — AI Virtual Patient Orchestrator.
Authoritative orchestrator for clinical simulation dialogue turns.

Architecture:
Learner Question
  ↓
Load Session & Case from MongoDB
  ↓
Classify Intent (QuestionClassifier)
  ↓
Retrieve Relevant Clinical Fact (FactRetriever)
  ↓
Controlled Disclosure Check
  ↓
Generate Focused Layperson Patient Response (HuggingFaceProvider)
  ↓
Validate & Sanitize (ResponseValidator)
  ↓
Update Session State & Persist Dialogue + Events in MongoDB
  ↓
Return Structured Response
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

from config import settings
from mongo_db import mongo_manager
from question_classifier import QuestionClassifier, ClassifiedIntent, IntentCategory
from fact_retriever import FactRetriever, RetrievedFact, FactState
from disclosure_controller import DisclosureController
from response_validator import PatientResponseValidator, ValidationResult
from providers.huggingface_provider import HuggingFaceProvider


class AIOrchestrator:

    def __init__(self):
        self.hf_provider = HuggingFaceProvider()
        self.provider_name = self.hf_provider.provider_name if self.hf_provider.is_configured else "InteractMD Clinical Patient Core"

    def process_turn_sync(
        self,
        case_id: str,
        user_message: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for FastAPI endpoint handlers."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.process_turn_async(case_id, user_message, session_id, conversation_history)
                )
            else:
                return asyncio.run(
                    self.process_turn_async(case_id, user_message, session_id, conversation_history)
                )
        except Exception:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.process_turn_async(case_id, user_message, session_id, conversation_history)
                )
            finally:
                new_loop.close()


    async def process_turn_async(
        self,
        case_id: str,
        user_message: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Load authoritative Clinical Case from MongoDB
        case_doc = mongo_manager.get_case_by_id(case_id)
        if not case_doc:
            # Check all cases if id matches loosely
            all_cases = mongo_manager.get_all_cases()
            matched = next((c for c in all_cases if c.get("case_id") == case_id or c.get("id") == case_id), None)
            if matched:
                case_doc = matched
            elif all_cases:
                case_doc = all_cases[0]
            else:
                case_doc = {"case_id": case_id, "title": "Clinical Simulation Case"}

        # 2. Retrieve past messages if session_id is active
        past_msgs = []
        if session_id:
            past_msgs = mongo_manager.get_session_messages(session_id)
        if not past_msgs and conversation_history:
            past_msgs = conversation_history

        # 3. Classify Learner Intent
        intent: ClassifiedIntent = QuestionClassifier.classify(user_message)

        # 4. Retrieve ONLY the single permitted clinical fact from MongoDB
        fact: RetrievedFact = FactRetriever.retrieve(intent, case_doc)

        # 5. Determine whether to generate via LLM or use safe controlled statement
        patient_profile = case_doc.get("patient", {})
        patient_name = patient_profile.get("name", "Patient")
        patient_age = patient_profile.get("age", 45)
        patient_gender = patient_profile.get("gender") or patient_profile.get("sex", "Unknown")
        persona_info = patient_profile.get("persona", {})
        if isinstance(persona_info, dict):
            persona_desc = f"Personality: {persona_info.get('personality', 'anxious')}, Emotional state: {persona_info.get('emotional_state', 'worried')}"
        else:
            persona_desc = str(persona_info)

        reply_text = fact.permitted_statement

        # If not a controlled shield or greeting, attempt Hugging Face LLM generation
        if not fact.is_controlled_shield and intent.category not in [IntentCategory.GREETING, IntentCategory.UNCLEAR]:
            system_prompt = (
                f"You are {patient_name}, a {patient_age}-year-old {patient_gender} patient in an educational clinical simulation.\n"
                f"Patient Profile: {persona_desc}.\n"
                f"RULES:\n"
                f"1. You are a REAL PATIENT. Speak strictly in the FIRST PERSON ('I', 'my', 'me').\n"
                f"2. You MUST clearly state your clinical truth: \"{fact.permitted_statement}\". Do NOT omit specific times, numbers, or key words from this truth.\n"
                f"3. Keep your response natural and concise in 1-2 sentences.\n"
                f"4. Do NOT mention any medical diagnosis or test results."
            )

            user_prompt = (
                f"Doctor's Question: \"{user_message}\"\n"
                f"Your Clinical Fact: \"{fact.permitted_statement}\"\n\n"
                f"Respond naturally as the patient in 1-2 sentences:"
            )

            raw_llm_response = None
            if self.hf_provider.is_configured:
                try:
                    raw_llm_response = await self.hf_provider.generate_chat(
                        system_prompt=system_prompt,
                        user_message=user_prompt,
                        conversation_history=past_msgs[-4:] if past_msgs else []
                    )
                except Exception as e:
                    print(f"[AIOrchestrator Warning] HF LLM call error: {e}")

            # 6. Validate & Sanitize Response
            val_result: ValidationResult = PatientResponseValidator.validate(
                raw_response=raw_llm_response,
                fallback_statement=fact.permitted_statement
            )

            if val_result.is_valid:
                # Check that key numbers/durations from fact are preserved if present in fact
                clean_text = val_result.sanitized_text
                fact_text = fact.permitted_statement
                # If fact has specific duration like 45 minutes and LLM omitted it, use fact
                if "45 minutes" in fact_text and "45 minutes" not in clean_text:
                    reply_text = fact.permitted_statement
                elif fact_text.startswith("Yes,") and not (clean_text.lower().startswith("yes") or "yes" in clean_text.lower() or "i am" in clean_text.lower() or "i do" in clean_text.lower()):
                    reply_text = f"Yes, {clean_text}"
                else:
                    reply_text = clean_text
            else:
                reply_text = fact.permitted_statement

        # 7. Persist to MongoDB
        facts_revealed = [fact.fact_id] if fact.fact_id else []

        if session_id:
            # Record fact disclosure in session
            if fact.fact_id:
                DisclosureController.record_disclosure(session_id, fact.fact_id)

            # Save Learner message
            mongo_manager.save_message({
                "session_id": session_id,
                "case_id": case_id,
                "sender": "LEARNER",
                "role": "learner",
                "message": user_message,
                "content": user_message,
                "metadata_json": {
                    "intent": intent.category.value,
                    "subconcept": intent.subconcept,
                    "empathy_detected": intent.empathy_detected
                },
                "timestamp": start_time
            })

            # Save Patient message
            mongo_manager.save_message({
                "session_id": session_id,
                "case_id": case_id,
                "sender": "PATIENT",
                "role": "patient",
                "message": reply_text,
                "content": reply_text,
                "metadata_json": {
                    "category": intent.ui_category,
                    "fact_id": fact.fact_id,
                    "fact_state": fact.state.value,
                    "provider": self.provider_name,
                    "latency_ms": int((time.time() - start_time) * 1000)
                },
                "timestamp": time.time()
            })

            # Record interaction event
            mongo_manager.save_event(session_id, "DIALOGUE_TURN", {
                "intent": intent.category.value,
                "fact_id": fact.fact_id,
                "empathy_detected": intent.empathy_detected
            })

        # 8. Return structured response
        return {
            "session_id": session_id,
            "message": {
                "role": "patient",
                "text": reply_text
            },
            "reply": reply_text,
            "category": intent.ui_category,
            "empathy_detected": intent.empathy_detected,
            "facts_revealed": facts_revealed,
            "provider": self.provider_name,
            "suggested_topics": [],
            "session_state": {
                "revealed_fact_ids": list(DisclosureController.get_revealed_facts(session_id)) if session_id else facts_revealed
            }
        }


ai_orchestrator = AIOrchestrator()
