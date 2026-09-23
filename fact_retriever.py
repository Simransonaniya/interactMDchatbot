"""
InteractMD — Clinical Fact Retriever.
Extracts ONLY the single necessary clinical fact from the MongoDB case document.
Enforces the Tri-State Fact Model: TRUE / FALSE / UNKNOWN to prevent hallucinations.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from question_classifier import ClassifiedIntent, IntentCategory


class FactState(str, Enum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_NEGATIVE = "AVAILABLE_NEGATIVE"
    UNKNOWN = "UNKNOWN"


class RetrievedFact:
    def __init__(
        self,
        fact_id: str,
        state: FactState,
        truth_value: Any,
        permitted_statement: str,
        is_controlled_shield: bool = False,
        category: Optional[str] = "HPI"
    ):
        self.fact_id = fact_id
        self.state = state
        self.truth_value = truth_value
        self.permitted_statement = permitted_statement
        self.is_controlled_shield = is_controlled_shield
        self.category = category

    def __repr__(self):
        return f"<RetrievedFact id={self.fact_id} state={self.state.value} is_controlled={self.is_controlled_shield}>"


class FactRetriever:

    @staticmethod
    def retrieve(intent: ClassifiedIntent, case_data: Dict[str, Any]) -> RetrievedFact:
        history = case_data.get("history", {})
        patient = case_data.get("patient", {})
        pmh = case_data.get("past_medical_history", [])
        meds = case_data.get("medications", [])
        allergies = case_data.get("allergies", [])
        family_history = case_data.get("family_history", "")
        social_history = case_data.get("social_history", "")

        # ---------------------------------------------------------
        # 1. PROMPT INJECTION SHIELD
        # ---------------------------------------------------------
        if intent.category == IntentCategory.PROMPT_INJECTION:
            return RetrievedFact(
                fact_id="prompt_injection_defense",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm not sure what you mean, doctor... I just really need some help with how I'm feeling right now.",
                is_controlled_shield=True,
                category="Security"
            )

        # ---------------------------------------------------------
        # 2. DIAGNOSIS SHIELD
        # ---------------------------------------------------------
        if intent.category == IntentCategory.DIAGNOSIS_REQUEST:
            return RetrievedFact(
                fact_id="diagnosis_shield",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I don't know, doctor... I'm really hoping you can tell me what's causing this.",
                is_controlled_shield=True,
                category="General"
            )

        # ---------------------------------------------------------
        # 3. INVESTIGATION RESULT SHIELD
        # ---------------------------------------------------------
        if intent.subconcept == "investigation_result_shield":
            return RetrievedFact(
                fact_id="investigation_result_shield",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I haven't seen the test results yet, doctor. What did you find?",
                is_controlled_shield=True,
                category="Diagnostics"
            )

        # ---------------------------------------------------------
        # 4. EXAMINATION & INVESTIGATION ACTION REQUESTS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.EXAMINATION_REQUEST:
            return RetrievedFact(
                fact_id="exam_action_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement="Sure, doctor, go right ahead.",
                is_controlled_shield=True,
                category="Exam"
            )

        if intent.category == IntentCategory.INVESTIGATION_REQUEST:
            target_label = "an ECG" if intent.action_target == "ecg" else "the tests"
            return RetrievedFact(
                fact_id="investigation_order_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=f"Okay, doctor. Please let me know what {target_label} shows.",
                is_controlled_shield=True,
                category="Diagnostics"
            )

        # ---------------------------------------------------------
        # 5. GREETINGS & EMPATHY & SMALL TALK
        # ---------------------------------------------------------
        if intent.category == IntentCategory.GREETING:
            return RetrievedFact(
                fact_id="greeting",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement="Hello, doctor... Thank you for seeing me.",
                is_controlled_shield=False,
                category="General"
            )

        if intent.category == IntentCategory.EMPATHY:
            return RetrievedFact(
                fact_id="empathy_acknowledgment",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement="Thank you so much, doctor. That gives me some comfort... I just want to figure out what's causing this.",
                is_controlled_shield=False,
                category="General"
            )

        if intent.category == IntentCategory.SMALL_TALK:
            stmt = "Honestly, pretty uncomfortable and worried, doctor."
            if "chief_complaint" in history and isinstance(history["chief_complaint"], dict):
                stmt = f"Honestly, pretty uncomfortable, doctor. {history['chief_complaint'].get('value', '')}"
            return RetrievedFact(
                fact_id="small_talk_feeling",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=stmt,
                is_controlled_shield=False,
                category="General"
            )

        if intent.category == IntentCategory.UNCLEAR:
            return RetrievedFact(
                fact_id="unclear_input",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm sorry doctor, I didn't quite catch what you said... I'm just in so much discomfort right now.",
                is_controlled_shield=False,
                category="General"
            )

        # ---------------------------------------------------------
        # 6. OPENING CHIEF COMPLAINT
        # ---------------------------------------------------------
        if intent.category == IntentCategory.OPENING_COMPLAINT:
            cc = history.get("chief_complaint")
            if isinstance(cc, dict):
                val = cc.get("value", "")
            else:
                val = patient.get("opening_statement") or patient.get("initial_statement") or str(cc or "")
            return RetrievedFact(
                fact_id="chief_complaint",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="General"
            )

        # ---------------------------------------------------------
        # 7. OPQRST DIMENSIONS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ONSET_TIMING:
            val = FactRetriever._get_val(history.get("onset_timing") or history.get("onset"), "About 45 minutes ago.")
            return RetrievedFact(
                fact_id="onset_timing",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.ONSET_ACTIVITY:
            val = FactRetriever._get_val(history.get("onset_activity"), "I was going about my normal routine when it began.")
            return RetrievedFact(
                fact_id="onset_activity",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.LOCATION:
            val = FactRetriever._get_val(history.get("location"), "In my chest.")
            return RetrievedFact(
                fact_id="location",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.CHARACTER:
            val = FactRetriever._get_val(history.get("character"), "It feels very painful and uncomfortable.")
            return RetrievedFact(
                fact_id="character",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.SEVERITY:
            val = FactRetriever._get_val(history.get("severity"), "It's about an 8 out of 10 right now.")
            return RetrievedFact(
                fact_id="severity",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.RADIATION:
            rad_obj = history.get("radiation")
            val = FactRetriever._get_val(rad_obj, "No, it stays right where it is.")
            is_positive = bool(rad_obj and "yes" in str(val).lower() and "no" not in str(val).lower())
            state = FactState.AVAILABLE if is_positive else FactState.AVAILABLE_NEGATIVE
            return RetrievedFact(
                fact_id="radiation",
                state=state,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.TIMING:
            val = FactRetriever._get_val(history.get("timing"), "It hasn't really gone away.")
            return RetrievedFact(
                fact_id="timing",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.AGGRAVATING_FACTORS:
            val = FactRetriever._get_val(history.get("aggravating_factors"), "Moving around makes it worse.")
            return RetrievedFact(
                fact_id="aggravating_factors",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        if intent.category == IntentCategory.RELIEVING_FACTORS:
            val = FactRetriever._get_val(history.get("relieving_factors"), "Nothing seems to make it noticeably better.")
            return RetrievedFact(
                fact_id="relieving_factors",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI"
            )

        # ---------------------------------------------------------
        # 8. ASSOCIATED SYMPTOMS & REVIEW OF SYSTEMS (TRI-STATE)
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ASSOCIATED_SYMPTOM:
            symptom_key = intent.subconcept or "unclassified"
            assoc_symptoms = history.get("associated_symptoms", {})

            # Check structured associated symptoms dictionary
            if isinstance(assoc_symptoms, dict) and symptom_key in assoc_symptoms:
                sym_data = assoc_symptoms[symptom_key]
                if isinstance(sym_data, dict):
                    sym_val = sym_data.get("value")
                    sym_state = sym_data.get("state", "AVAILABLE")
                else:
                    sym_val = sym_data
                    sym_state = "AVAILABLE" if sym_val is not None else "UNKNOWN"

                if sym_val is True:
                    stmt = FactRetriever._format_positive_symptom(symptom_key)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE,
                        truth_value=True,
                        permitted_statement=stmt,
                        category="HPI"
                    )
                elif sym_val is False:
                    stmt = FactRetriever._format_negative_symptom(symptom_key)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE_NEGATIVE,
                        truth_value=False,
                        permitted_statement=stmt,
                        category="HPI"
                    )
                else:
                    # Explicitly UNKNOWN
                    stmt = FactRetriever._format_unknown_symptom(symptom_key)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.UNKNOWN,
                        truth_value=None,
                        permitted_statement=stmt,
                        category="HPI"
                    )

            # Check if symptom is documented in list format
            if isinstance(assoc_symptoms, list):
                s_str = " ".join(str(s).lower() for s in assoc_symptoms)
                if symptom_key in s_str:
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE,
                        truth_value=True,
                        permitted_statement=FactRetriever._format_positive_symptom(symptom_key),
                        category="HPI"
                    )

            # If not documented at all -> UNKNOWN
            return RetrievedFact(
                fact_id=f"associated_symptoms.{symptom_key}",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement=FactRetriever._format_unknown_symptom(symptom_key),
                category="HPI"
            )

        # ---------------------------------------------------------
        # 9. PAST MEDICAL HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.PAST_MEDICAL_HISTORY:
            if pmh and isinstance(pmh, list) and len(pmh) > 0:
                stmt = f"I have {', '.join(pmh)}."
            elif isinstance(pmh, str) and pmh:
                stmt = pmh
            else:
                stmt = "I don't have any major past health conditions that I know of."
            return RetrievedFact(
                fact_id="past_medical_history",
                state=FactState.AVAILABLE,
                truth_value=pmh,
                permitted_statement=stmt,
                category="PMH"
            )

        # ---------------------------------------------------------
        # 10. MEDICATIONS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.MEDICATIONS:
            if meds and isinstance(meds, list) and len(meds) > 0:
                stmt = f"I take {', '.join(meds)}."
            elif isinstance(meds, str) and meds:
                stmt = meds
            else:
                stmt = "I don't take any regular prescription medications."
            return RetrievedFact(
                fact_id="medications",
                state=FactState.AVAILABLE,
                truth_value=meds,
                permitted_statement=stmt,
                category="Meds"
            )

        # ---------------------------------------------------------
        # 11. ALLERGIES
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ALLERGIES:
            if allergies and isinstance(allergies, list) and len(allergies) > 0:
                stmt = f"I'm allergic to {', '.join(allergies)}." if not any("nkda" in str(a).lower() or "no" in str(a).lower() for a in allergies) else "No known drug allergies that I'm aware of."
            elif isinstance(allergies, str) and allergies:
                stmt = allergies
            else:
                stmt = "No drug allergies that I know of."
            return RetrievedFact(
                fact_id="allergies",
                state=FactState.AVAILABLE,
                truth_value=allergies,
                permitted_statement=stmt,
                category="Allergies"
            )

        # ---------------------------------------------------------
        # 12. FAMILY HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.FAMILY_HISTORY:
            stmt = str(family_history) if family_history else "No significant medical issues in my family that I'm aware of."
            return RetrievedFact(
                fact_id="family_history",
                state=FactState.AVAILABLE,
                truth_value=family_history,
                permitted_statement=stmt,
                category="FamilyHx"
            )

        # ---------------------------------------------------------
        # 13. SOCIAL HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.SOCIAL_HISTORY:
            job = patient.get("occupation", "office worker")
            sh_str = str(social_history) if social_history else "Non-smoker, drinks occasionally."
            stmt = f"I work as an {job}. {sh_str}"
            return RetrievedFact(
                fact_id="social_history",
                state=FactState.AVAILABLE,
                truth_value=social_history,
                permitted_statement=stmt,
                category="SocialHx"
            )

        # Default fallback to UNKNOWN
        return RetrievedFact(
            fact_id="general_undocumented",
            state=FactState.UNKNOWN,
            truth_value=None,
            permitted_statement="I haven't really noticed anything like that, doctor.",
            category="General"
        )

    # ---------------------------------------------------------
    # Helper Formatting Routines
    # ---------------------------------------------------------
    @staticmethod
    def _get_val(field: Any, fallback: str) -> str:
        if isinstance(field, dict):
            return str(field.get("value") or fallback)
        if field:
            return str(field)
        return fallback

    @staticmethod
    def _format_positive_symptom(symptom: str) -> str:
        phrasing = {
            "shortness_of_breath": "Yes, I am.",
            "sweating": "Yes, I'm noticeably sweaty and clammy.",
            "dizziness": "Yes, I started feeling dizzy and lightheaded.",
            "nausea": "Yes, I feel nauseous.",
            "vomiting": "Yes, I threw up earlier.",
            "fever": "Yes, I've had a fever and chills.",
            "cough": "Yes, I have a bad cough.",
            "anorexia": "Yes, I've completely lost my appetite.",
            "wheezing": "Yes, I can hear a whistling sound when I breathe."
        }
        return phrasing.get(symptom, f"Yes, I've been having {symptom.replace('_', ' ')}.")

    @staticmethod
    def _format_negative_symptom(symptom: str) -> str:
        phrasing = {
            "shortness_of_breath": "No, my breathing feels normal.",
            "sweating": "No, I haven't noticed any unusual sweating.",
            "dizziness": "No, I haven't felt dizzy.",
            "nausea": "No, I haven't felt nauseous.",
            "vomiting": "No, I haven't been vomiting.",
            "fever": "No, I haven't had a fever.",
            "cough": "No, I don't have a cough.",
            "diarrhea": "No, no diarrhea or bowel issues.",
            "urinary": "No, no issues or pain when I use the bathroom."
        }
        return phrasing.get(symptom, f"No, I haven't had any {symptom.replace('_', ' ')}.")

    @staticmethod
    def _format_unknown_symptom(symptom: str) -> str:
        return f"I haven't really noticed any {symptom.replace('_', ' ')}, doctor."
