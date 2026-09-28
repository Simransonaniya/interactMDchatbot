"""
InteractMD — AI Virtual Patient Orchestrator.
Authoritative orchestrator for clinical simulation dialogue turns.

Architecture:
Learner Question
  ↓
Load Session & Case from MongoDB / Relational Storage
  ↓
Build / Update PatientSimulationState (with Authoritative Revealed Facts)
  ↓
Classify Intent (QuestionClassifier: HISTORY_QUESTION, CLARIFICATION, CONFIRMATION, EMPATHY_REASSURANCE, MANAGEMENT_STATEMENT, MEDICATION_STATEMENT, DIAGNOSIS_STATEMENT, EXAM_REQUEST, INVESTIGATION_REQUEST, OFF_TOPIC, UNKNOWN)
  ↓
Retrieve Relevant Clinical Fact (FactRetriever - Grounded with Session State Memory)
  ↓
Controlled Disclosure & Closed-World Gate
  ↓
Generate Focused Layperson Patient Response (HuggingFaceProvider) / Direct Grounded Statement
  ↓
Validate & Sanitize (ResponseValidator - Contradiction Checks)
  ↓
Update Session State & Persist Dialogue + Events in MongoDB
  ↓
Return Structured Response with Internal Fact Tracking
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

from config import settings
from mongo_db import mongo_manager
from patient_state import PatientSimulationState, PatientStateManager, FactState
from question_classifier import QuestionClassifier, ClassifiedIntent, IntentCategory
from fact_retriever import FactRetriever, RetrievedFact
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

        # 1. Load authoritative Clinical Case from MongoDB / Storage
        case_doc = mongo_manager.get_case_by_id(case_id)
        if not case_doc:
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

        # 3. Build / Synchronize Active Patient Simulation State
        session_state = PatientStateManager.get_or_create(
            case_id=case_id,
            session_id=session_id,
            case_data=case_doc,
            conversation_history=past_msgs
        )

        session_state.last_clinician_message = user_message

        # 4. Classify Learner Intent (with Previous-Turn Context)
        intent: ClassifiedIntent = QuestionClassifier.classify(user_message, session_state=session_state)
        session_state.last_intent = intent.category.value
        session_state.last_topic = intent.subconcept or intent.category.value
        session_state.last_question_topic = session_state.last_topic
        session_state.last_time_reference = intent.time_reference

        # 5. Retrieve Relevant Clinical Fact (Memory & Case Grounded)
        fact: RetrievedFact = FactRetriever.retrieve(
            intent=intent,
            case_data=case_doc,
            session_state=session_state
        )

        # 6. Closed-World Decision & Response Phrasing
        patient_name = session_state.demographics.get("name", "Patient")
        patient_age = session_state.demographics.get("age", 45)
        patient_gender = session_state.demographics.get("gender", "Unknown")

        reply_text = fact.permitted_statement

        # Determine if LLM rephrasing is permitted
        deterministic_intents = [
            IntentCategory.GREETING,
            IntentCategory.UNCLEAR,
            IntentCategory.UNKNOWN,
            IntentCategory.OUT_OF_SCOPE,
            IntentCategory.OFF_TOPIC,
            IntentCategory.CHALLENGE,
            IntentCategory.CLARIFICATION,
            IntentCategory.CONFIRMATION,
            IntentCategory.MANAGEMENT_STATEMENT,
            IntentCategory.MEDICATION_STATEMENT,
            IntentCategory.MEDICATION_NAME_FRAGMENT,
            IntentCategory.DIAGNOSIS_STATEMENT,
            IntentCategory.DIAGNOSIS_REQUEST,
            IntentCategory.EMPATHY,
            IntentCategory.EMPATHY_REASSURANCE,
            IntentCategory.EXAMINATION_REQUEST,
            IntentCategory.EXAM_REQUEST,
            IntentCategory.INVESTIGATION_REQUEST,
            IntentCategory.GENDER_INAPPLICABLE
        ]

        should_use_llm = (
            not fact.is_controlled_shield
            and not fact.is_previously_revealed
            and fact.state in [FactState.AVAILABLE, FactState.AVAILABLE_NEGATIVE]
            and intent.category not in deterministic_intents
        )

        if should_use_llm and self.hf_provider.is_configured:
            neg_instruction = ""
            if fact.permitted_statement.startswith("No,") or fact.state == FactState.AVAILABLE_NEGATIVE:
                neg_instruction = f" You DO NOT have this symptom. You MUST state: \"{fact.permitted_statement}\"."

            revealed_summary = "; ".join([f"{k}: {v}" for k, v in list(session_state.revealed_facts.items())[:4]])

            system_prompt = (
                f"You are {patient_name}, a {patient_age}-year-old {patient_gender} patient in an emergency medical encounter.\n"
                f"RULES:\n"
                f"1. You are a REAL PATIENT. Speak strictly in the FIRST PERSON ('I', 'my', 'me').\n"
                f"2. You MUST strictly stick to your authorized clinical fact: \"{fact.permitted_statement}\".{neg_instruction}\n"
                f"3. Previously revealed facts: {revealed_summary if revealed_summary else 'None yet'}.\n"
                f"4. Do NOT invent diagnoses, background activities, unmentioned medications, or new symptoms.\n"
                f"5. Answer directly and naturally in 1-2 concise sentences."
            )

            user_prompt = (
                f"Doctor's Question: \"{user_message}\"\n"
                f"Your Clinical Fact: \"{fact.permitted_statement}\"\n\n"
                f"State this clinical fact directly and naturally as the patient in 1-2 sentences:"
            )

            raw_llm_response = None
            try:
                raw_llm_response = await self.hf_provider.generate_chat(
                    system_prompt=system_prompt,
                    user_message=user_prompt,
                    conversation_history=past_msgs[-4:] if past_msgs else []
                )
            except Exception as e:
                print(f"[AIOrchestrator Warning] LLM call error: {e}")

            # 7. Validate & Sanitize Response
            val_result: ValidationResult = PatientResponseValidator.validate(
                raw_response=raw_llm_response,
                fallback_statement=fact.permitted_statement,
                session_state=session_state
            )

            if val_result.is_valid:
                clean_text = val_result.sanitized_text
                fact_text = fact.permitted_statement
                lower_clean = clean_text.lower()
                lower_fact = fact_text.lower()

                # Verify vital clinical keywords were not omitted by LLM rephrasing
                if intent.category == IntentCategory.ONSET_TIMING and "45" in fact_text and "45" not in clean_text:
                    reply_text = fact_text
                elif intent.category == IntentCategory.SEVERITY and "8" in fact_text and "8" not in clean_text:
                    reply_text = fact_text
                elif intent.category == IntentCategory.RADIATION and ("jaw" in lower_fact or "arm" in lower_fact) and not ("jaw" in lower_clean or "arm" in lower_clean):
                    reply_text = fact_text
                elif intent.category == IntentCategory.CHARACTER and any(k in lower_fact for k in ["elephant", "pressure", "crushing", "squeezing", "heavy"]) and not any(k in lower_clean for k in ["elephant", "pressure", "crushing", "squeezing", "heavy", "chest"]):
                    reply_text = fact_text
                elif fact_text.startswith("No,") or fact.state == FactState.AVAILABLE_NEGATIVE:
                    if not any(k in lower_clean for k in ["no", "not", "haven't", "don't", "never", "none"]):
                        reply_text = fact_text
                    elif not lower_clean.startswith("no"):
                        reply_text = f"No, {clean_text}"
                    else:
                        reply_text = clean_text
                elif fact_text.startswith("Yes,") and not any(k in clean_text.lower() for k in ["yes", "i am", "i do", "i have", "definitely"]):
                    reply_text = f"Yes, {clean_text}"
                else:
                    reply_text = clean_text
            else:
                reply_text = fact.permitted_statement
        else:
            reply_text = fact.permitted_statement

        # 8. Update Session State with Revealed Fact and Last Patient Statement
        session_state.last_patient_message = reply_text
        session_state.last_patient_fact_state = fact.state.value
        session_state.last_slot = (intent.slots[0] if intent.slots else fact.fact_key) or intent.subconcept
        if fact.fact_id:
            session_state.record_disclosure(fact.fact_id, reply_text)
            if fact.fact_key:
                session_state.record_disclosure(fact.fact_key, reply_text)

        # 9. Persist to MongoDB / Relational DB
        facts_revealed = list(session_state.revealed_fact_ids)

        if session_id:
            if fact.fact_id:
                DisclosureController.record_disclosure(session_id, fact.fact_id)

            mongo_manager.save_message({
                "session_id": session_id,
                "case_id": case_id,
                "sender": "LEARNER",
                "role": "learner",
                "message": user_message,
                "content": user_message,
                "metadata_json": {
                    "intent": intent.category.value,
                    "primary_type": intent.primary_type,
                    "subconcept": intent.subconcept,
                    "empathy_detected": intent.empathy_detected,
                    "is_follow_up_to_previous_turn": intent.is_follow_up_to_previous_turn,
                    "relationship": intent.relationship,
                    "referenced_topic": intent.referenced_topic,
                    "referenced_slot": intent.referenced_slot
                },
                "timestamp": start_time
            })

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
                    "response_source": fact.response_source,
                    "fact_key": fact.fact_key,
                    "provider": self.provider_name,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "is_follow_up_to_previous_turn": intent.is_follow_up_to_previous_turn,
                    "relationship": intent.relationship
                },
                "timestamp": time.time()
            })

            mongo_manager.save_event(session_id, "DIALOGUE_TURN", {
                "intent": intent.category.value,
                "primary_type": intent.primary_type,
                "fact_id": fact.fact_id,
                "response_source": fact.response_source,
                "empathy_detected": intent.empathy_detected,
                "relationship": intent.relationship
            })

        # 10. Return Structured Response
        return {
            "session_id": session_id,
            "message": {
                "role": "patient",
                "text": reply_text
            },
            "reply": reply_text,
            "category": intent.ui_category,
            "intent": intent.primary_type,
            "intent_category": intent.category.value,
            "is_follow_up_to_previous_turn": intent.is_follow_up_to_previous_turn,
            "relationship": intent.relationship,
            "referenced_topic": intent.referenced_topic,
            "referenced_slot": intent.referenced_slot,
            "empathy_detected": intent.empathy_detected,
            "facts_revealed": facts_revealed,
            "response_source": fact.response_source,
            "fact_key": fact.fact_key,
            "provider": self.provider_name,
            "suggested_topics": [],
            "session_state": {
                "revealed_fact_ids": facts_revealed,
                "last_clinician_message": user_message,
                "last_patient_message": reply_text,
                "last_intent": intent.category.value,
                "last_topic": session_state.last_topic,
                "last_slot": session_state.last_slot,
                "last_time_reference": session_state.last_time_reference,
                "last_patient_fact_state": fact.state.value
            }
        }


ai_orchestrator = AIOrchestrator()
