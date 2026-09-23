"""
InteractMD — Patient Dialogue Generator.
Generates focused, empathetic, 1st-person patient dialogue grounded ONLY in the single relevant fact.
"""

from typing import Dict, Any, List, Optional
from question_analyzer import AnalyzedQuestion, QuestionCategory
from fact_selector import FactResult
from patient_state import PatientSessionState
from response_validator import PatientResponseValidator
from llm_client import llm_client


class PatientDialogueGenerator:

    @staticmethod
    def generate(
        analyzed: AnalyzedQuestion,
        fact_result: FactResult,
        patient_profile: Dict[str, Any],
        session_state: PatientSessionState
    ) -> str:
        # If the result is a controlled action (e.g. ECG shield, prompt injection defense, exam confirmation),
        # return the verified safe natural statement directly.
        if fact_result.is_controlled:
            return fact_result.natural_patient_statement

        # For greetings, small talk, or bedside empathy acknowledgment, return natural statement
        if analyzed.category in [QuestionCategory.NON_MEDICAL_OR_SMALL_TALK, QuestionCategory.QUESTION_UNCLEAR]:
            return fact_result.natural_patient_statement

        # Construct concise prompt for LLM containing ONLY the allowed single fact
        name = patient_profile.get("name", "Patient")
        age = patient_profile.get("age", 50)
        gender = patient_profile.get("gender", "Unknown")
        persona = patient_profile.get("persona", "Anxious, uncomfortable")
        emotional_state = session_state.emotional_state

        system_prompt = (
            f"You are {name}, a {age}-year-old {gender} patient in a medical simulation encounter.\n"
            f"Persona & Tone: {persona}. Current emotional state: {emotional_state}.\n"
            f"CRITICAL CLINICAL RULES:\n"
            f"1. You are a REAL PATIENT speaking to a doctor. Always speak in FIRST PERSON ('I', 'my', 'me').\n"
            f"2. Never use 3rd-person pronouns ('his', 'her', 'the patient').\n"
            f"3. Answer ONLY the doctor's specific question using the provided fact. Keep it to 1-2 concise sentences.\n"
            f"4. Do NOT dump extra medical history, other symptoms, or case summaries.\n"
            f"5. Do NOT use clinical jargon or diagnose yourself.\n"
            f"6. Do NOT ask interview-driving questions like 'what else do you need to know?'."
        )

        user_prompt = (
            f"Doctor's Question: \"{analyzed.raw_text}\"\n"
            f"Your Knowledge Fact: \"{fact_result.natural_patient_statement}\"\n\n"
            f"Respond naturally as the patient in 1-2 sentences:"
        )

        # Attempt generation if external LLM configured, else use verified natural statement
        raw_response = None
        # If API keys are active, call LLM client synchronously or use fallback
        if llm_client.openai_key or llm_client.groq_key or llm_client.gemini_key:
            try:
                import asyncio
                # Try running async call with short timeout if loop is available
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop and loop.is_running():
                    # If already in an async event loop, create task or use fallback
                    raw_response = fact_result.natural_patient_statement
                else:
                    raw_response = loop.run_until_complete(
                        llm_client.generate_response(
                            system_prompt=system_prompt,
                            user_message=user_prompt,
                            conversation_history=session_state.get_recent_history(2),
                            retrieved_context=fact_result.natural_patient_statement
                        )
                    )
            except Exception as e:
                # Silently fallback to high-fidelity natural statement
                raw_response = fact_result.natural_patient_statement

        if not raw_response:
            raw_response = fact_result.natural_patient_statement

        # Validate and sanitize response before returning
        clean_response = PatientResponseValidator.validate_and_sanitize(
            raw_response=raw_response,
            expected_facet=fact_result.concept,
            fallback_statement=fact_result.natural_patient_statement
        )

        return clean_response
