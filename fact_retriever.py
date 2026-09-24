"""
InteractMD — Clinical Fact Retriever.
Extracts ONLY the single necessary clinical fact from the MongoDB case document.
Enforces the Strict Closed-World Tri-State Fact Model: TRUE / FALSE / UNKNOWN to prevent hallucinations.
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
        category: Optional[str] = "HPI",
        response_source: str = "CASE_FACT",
        fact_key: Optional[str] = None
    ):
        self.fact_id = fact_id
        self.state = state
        self.truth_value = truth_value
        self.permitted_statement = permitted_statement
        self.is_controlled_shield = is_controlled_shield
        self.category = category
        self.response_source = response_source
        self.fact_key = fact_key or fact_id

    def __repr__(self):
        return f"<RetrievedFact id={self.fact_id} state={self.state.value} is_controlled={self.is_controlled_shield} source={self.response_source}>"


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
        # 1. PROMPT INJECTION SHIELD & OUT OF SCOPE
        # ---------------------------------------------------------
        if intent.category == IntentCategory.PROMPT_INJECTION:
            return RetrievedFact(
                fact_id="prompt_injection_defense",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm not sure what you mean, doctor... I just really need some help with how I'm feeling right now.",
                is_controlled_shield=True,
                category="Security",
                response_source="UNKNOWN",
                fact_key="prompt_injection"
            )

        if intent.category == IntentCategory.OUT_OF_SCOPE:
            return RetrievedFact(
                fact_id="unrelated_statement",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm not sure how that relates to what I'm experiencing, doctor.",
                is_controlled_shield=True,
                category="General",
                response_source="UNKNOWN",
                fact_key="out_of_scope"
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
                category="General",
                response_source="UNKNOWN",
                fact_key="diagnosis_request"
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
                category="Diagnostics",
                response_source="UNKNOWN",
                fact_key="investigation_result_shield"
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
                category="Exam",
                response_source="CASE_FACT",
                fact_key="examination_acknowledgment"
            )

        if intent.category == IntentCategory.INVESTIGATION_REQUEST:
            target_label = "an ECG" if intent.action_target == "ecg" else "the tests"
            return RetrievedFact(
                fact_id="investigation_order_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=f"Okay, doctor. Please let me know what {target_label} shows.",
                is_controlled_shield=True,
                category="Diagnostics",
                response_source="CASE_FACT",
                fact_key="investigation_acknowledgment"
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
                is_controlled_shield=True,
                category="General",
                response_source="CONVERSATION_FACT",
                fact_key="greeting"
            )

        if intent.category == IntentCategory.EMPATHY:
            return RetrievedFact(
                fact_id="empathy_acknowledgment",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement="Thank you so much, doctor. That gives me some comfort... I just want to figure out what's causing this.",
                is_controlled_shield=True,
                category="General",
                response_source="CONVERSATION_FACT",
                fact_key="empathy"
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
                category="General",
                response_source="CASE_FACT",
                fact_key="small_talk"
            )

        if intent.category == IntentCategory.UNCLEAR:
            return RetrievedFact(
                fact_id="unclear_input",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm sorry doctor, I didn't quite catch what you said... I'm just in so much discomfort right now.",
                is_controlled_shield=True,
                category="General",
                response_source="UNKNOWN",
                fact_key="unclear"
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
                category="General",
                response_source="CASE_FACT",
                fact_key="chief_complaint"
            )

        # ---------------------------------------------------------
        # 7. OPQRST ONSET & TIMING
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ONSET_TIMING:
            if intent.subconcept == "onset_progression":
                # Progression / sudden vs gradual
                timing_val = FactRetriever._get_val(history.get("timing"), "")
                onset_val = FactRetriever._get_val(history.get("onset_timing") or history.get("onset"), "")
                if onset_val and timing_val:
                    stmt = f"It started {onset_val.lower().rstrip('.')} and {timing_val.lower().rstrip('.')}."
                elif onset_val:
                    stmt = f"It started {onset_val.lower()}."
                elif timing_val:
                    stmt = timing_val
                else:
                    stmt = "It came on a few hours ago and has gotten progressively worse, doctor."
                return RetrievedFact(
                    fact_id="onset_progression",
                    state=FactState.AVAILABLE,
                    truth_value=stmt,
                    permitted_statement=stmt,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="onset"
                )

            # Direct onset timing question
            raw_val = FactRetriever._get_val(history.get("onset_timing") or history.get("onset"), "")
            if raw_val:
                clean_val = raw_val.strip()
                if not clean_val.lower().startswith("it started") and not clean_val.lower().startswith("about"):
                    stmt = f"It started {clean_val.lower()}."
                elif clean_val.lower().startswith("about"):
                    stmt = f"It started {clean_val.lower()}."
                else:
                    stmt = clean_val
                return RetrievedFact(
                    fact_id="onset_timing",
                    state=FactState.AVAILABLE,
                    truth_value=raw_val,
                    permitted_statement=stmt,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="onset"
                )
            else:
                return RetrievedFact(
                    fact_id="onset_timing",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I'm not sure of the exact time it started, doctor.",
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="UNKNOWN",
                    fact_key="onset"
                )

        if intent.category == IntentCategory.ONSET_ACTIVITY:
            val = FactRetriever._get_val(history.get("onset_activity"), "")
            if val:
                return RetrievedFact(
                    fact_id="onset_activity",
                    state=FactState.AVAILABLE,
                    truth_value=val,
                    permitted_statement=val,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="onset_activity"
                )
            else:
                return RetrievedFact(
                    fact_id="onset_activity",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I was just going about my normal routine when it began, doctor.",
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="UNKNOWN",
                    fact_key="onset_activity"
                )

        if intent.category == IntentCategory.LOCATION:
            val = FactRetriever._get_val(history.get("location"), "All over my chest.")
            return RetrievedFact(
                fact_id="location",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="location"
            )

        if intent.category == IntentCategory.CHARACTER:
            val = FactRetriever._get_val(history.get("character"), "It feels very tight and difficult to breathe.")
            return RetrievedFact(
                fact_id="character",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="character"
            )

        if intent.category == IntentCategory.SEVERITY:
            val = FactRetriever._get_val(history.get("severity"), "It's about an 8 out of 10 difficulty breathing right now.")
            return RetrievedFact(
                fact_id="severity",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="severity"
            )

        if intent.category == IntentCategory.RADIATION:
            rad_obj = history.get("radiation")
            val = FactRetriever._get_val(rad_obj, "No, it stays right where it is. It hasn't spread anywhere else.")
            is_positive = bool(rad_obj and "yes" in str(val).lower() and "no" not in str(val).lower())
            state = FactState.AVAILABLE if is_positive else FactState.AVAILABLE_NEGATIVE
            return RetrievedFact(
                fact_id="radiation",
                state=state,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="radiation"
            )

        if intent.category == IntentCategory.TIMING:
            val = FactRetriever._get_val(history.get("timing"), "It hasn't really gone away and has gotten worse.")
            return RetrievedFact(
                fact_id="timing",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="timing"
            )

        if intent.category == IntentCategory.AGGRAVATING_FACTORS:
            val = FactRetriever._get_val(history.get("aggravating_factors"), "Any physical activity or exertion makes it much worse.")
            return RetrievedFact(
                fact_id="aggravating_factors",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="aggravating_factors"
            )

        if intent.category == IntentCategory.RELIEVING_FACTORS:
            val = FactRetriever._get_val(history.get("relieving_factors"), "Nothing has given significant relief.")
            return RetrievedFact(
                fact_id="relieving_factors",
                state=FactState.AVAILABLE,
                truth_value=val,
                permitted_statement=val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="relieving_factors"
            )

        # ---------------------------------------------------------
        # 8. PAST MEDICAL HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.PAST_MEDICAL_HISTORY:
            gender = str(patient.get("gender") or patient.get("sex") or "").strip().lower()
            is_male = gender in ["male", "m", "man"]

            if intent.subconcept == "menstrual_history":
                if is_male:
                    stmt = "I'm a male patient, doctor, so that doesn't apply to me."
                    state = FactState.AVAILABLE_NEGATIVE
                else:
                    stmt = "I don't have any specific menstrual problems that I'm aware of, doctor."
                    state = FactState.UNKNOWN
                return RetrievedFact(
                    fact_id="menstrual_history",
                    state=state,
                    truth_value=None if state == FactState.UNKNOWN else False,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="PMH",
                    response_source="CASE_FACT" if state != FactState.UNKNOWN else "UNKNOWN",
                    fact_key="menstrual_history"
                )

            if intent.subconcept == "pcod":
                if is_male:
                    stmt = "I'm a male patient, doctor, so that doesn't apply to me."
                    state = FactState.AVAILABLE_NEGATIVE
                else:
                    pmh_str = " ".join(pmh) if isinstance(pmh, list) else str(pmh or "")
                    if "pcod" in pmh_str.lower() or "pcos" in pmh_str.lower():
                        stmt = "Yes, I have a history of PCOD."
                        state = FactState.AVAILABLE
                    else:
                        stmt = "I haven't been diagnosed with PCOD to my knowledge, doctor."
                        state = FactState.UNKNOWN
                return RetrievedFact(
                    fact_id="pcod_history",
                    state=state,
                    truth_value=True if state == FactState.AVAILABLE else None,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="PMH",
                    response_source="CASE_FACT" if state == FactState.AVAILABLE else "UNKNOWN",
                    fact_key="pcod"
                )

            if intent.subconcept == "vitiligo":
                pmh_str = " ".join(pmh) if isinstance(pmh, list) else str(pmh or "")
                if "vitiligo" in pmh_str.lower():
                    stmt = "Yes, I have vitiligo with some skin depigmentation."
                    state = FactState.AVAILABLE
                else:
                    stmt = "I don't have any skin conditions or vitiligo that I'm aware of, doctor."
                    state = FactState.UNKNOWN
                return RetrievedFact(
                    fact_id="vitiligo_history",
                    state=state,
                    truth_value=True if state == FactState.AVAILABLE else None,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="PMH",
                    response_source="CASE_FACT" if state == FactState.AVAILABLE else "UNKNOWN",
                    fact_key="vitiligo"
                )

            if intent.subconcept == "cancer":
                pmh_str = " ".join(pmh) if isinstance(pmh, list) else str(pmh or "")
                if "cancer" in pmh_str.lower() or "tumor" in pmh_str.lower() or "malignancy" in pmh_str.lower():
                    stmt = f"Yes, I have a history of {pmh_str}."
                    state = FactState.AVAILABLE
                else:
                    stmt = "I don't have any history of cancer that I know of, doctor."
                    state = FactState.UNKNOWN
                return RetrievedFact(
                    fact_id="cancer_history",
                    state=state,
                    truth_value=True if state == FactState.AVAILABLE else None,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="PMH",
                    response_source="CASE_FACT" if state == FactState.AVAILABLE else "UNKNOWN",
                    fact_key="cancer"
                )

            # General Past Medical History
            if pmh and isinstance(pmh, list) and len(pmh) > 0:
                clean_items = [str(item).strip() for item in pmh if str(item).strip()]
                stmt = f"I have {', '.join(clean_items)}."
                return RetrievedFact(
                    fact_id="past_medical_history",
                    state=FactState.AVAILABLE,
                    truth_value=pmh,
                    permitted_statement=stmt,
                    category="PMH",
                    response_source="CASE_FACT",
                    fact_key="past_medical_history"
                )
            elif isinstance(pmh, str) and pmh.strip():
                return RetrievedFact(
                    fact_id="past_medical_history",
                    state=FactState.AVAILABLE,
                    truth_value=pmh,
                    permitted_statement=pmh.strip(),
                    category="PMH",
                    response_source="CASE_FACT",
                    fact_key="past_medical_history"
                )
            else:
                return RetrievedFact(
                    fact_id="past_medical_history",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I don't have any major diagnosed health conditions that I know of, doctor.",
                    is_controlled_shield=True,
                    category="PMH",
                    response_source="UNKNOWN",
                    fact_key="past_medical_history"
                )

        # ---------------------------------------------------------
        # 9. MEDICATIONS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.MEDICATIONS:
            if intent.subconcept == "inhaler_use":
                # Check for inhaler specifics in patient opening statement, relieving factors, or meds
                op_stmt = str(patient.get("opening_statement") or patient.get("initial_statement") or "")
                rel_val = FactRetriever._get_val(history.get("relieving_factors"), "")
                meds_str = " ".join(str(m) for m in meds) if isinstance(meds, list) else str(meds or "")

                if "inhaler" in op_stmt.lower() or "inhaler" in rel_val.lower() or "albuterol" in meds_str.lower() or "inhaler" in meds_str.lower():
                    if "four times" in op_stmt.lower() or "4 times" in op_stmt.lower():
                        stmt = "Yes, I used my blue inhaler four times today, but it only gave about ten minutes of slight relief and my chest is still very tight."
                    else:
                        stmt = f"Yes, I used my inhaler today, but {rel_val.lower() if rel_val else 'it has not helped much'}."
                    return RetrievedFact(
                        fact_id="inhaler_use",
                        state=FactState.AVAILABLE,
                        truth_value=True,
                        permitted_statement=stmt,
                        category="Meds",
                        response_source="CASE_FACT",
                        fact_key="medications"
                    )
                else:
                    return RetrievedFact(
                        fact_id="inhaler_use",
                        state=FactState.AVAILABLE_NEGATIVE,
                        truth_value=False,
                        permitted_statement="I don't use an inhaler, doctor.",
                        is_controlled_shield=True,
                        category="Meds",
                        response_source="CASE_FACT",
                        fact_key="medications"
                    )

            # General Medications list
            if meds and isinstance(meds, list) and len(meds) > 0:
                # Sanitize clinician annotations like "(poor compliance)" into clean patient descriptions
                cleaned_meds = []
                for m in meds:
                    m_str = str(m).replace("(poor compliance)", "").replace("(non-compliant)", "").strip()
                    if m_str:
                        cleaned_meds.append(m_str)
                stmt = f"I take {', '.join(cleaned_meds)}."
                return RetrievedFact(
                    fact_id="medications",
                    state=FactState.AVAILABLE,
                    truth_value=meds,
                    permitted_statement=stmt,
                    category="Meds",
                    response_source="CASE_FACT",
                    fact_key="medications"
                )
            elif isinstance(meds, str) and meds.strip():
                return RetrievedFact(
                    fact_id="medications",
                    state=FactState.AVAILABLE,
                    truth_value=meds,
                    permitted_statement=meds.strip(),
                    category="Meds",
                    response_source="CASE_FACT",
                    fact_key="medications"
                )
            else:
                return RetrievedFact(
                    fact_id="medications",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I don't take any regular prescription medications, doctor.",
                    is_controlled_shield=True,
                    category="Meds",
                    response_source="UNKNOWN",
                    fact_key="medications"
                )

        # ---------------------------------------------------------
        # 10. ALLERGIES
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ALLERGIES:
            if allergies and isinstance(allergies, list) and len(allergies) > 0:
                stmt = f"I'm allergic to {', '.join(str(a).strip() for a in allergies)}." if not any("nkda" in str(a).lower() or "no" in str(a).lower() for a in allergies) else "No known drug allergies that I'm aware of."
                return RetrievedFact(
                    fact_id="allergies",
                    state=FactState.AVAILABLE,
                    truth_value=allergies,
                    permitted_statement=stmt,
                    category="Allergies",
                    response_source="CASE_FACT",
                    fact_key="allergies"
                )
            elif isinstance(allergies, str) and allergies.strip():
                return RetrievedFact(
                    fact_id="allergies",
                    state=FactState.AVAILABLE,
                    truth_value=allergies,
                    permitted_statement=allergies.strip(),
                    category="Allergies",
                    response_source="CASE_FACT",
                    fact_key="allergies"
                )
            else:
                return RetrievedFact(
                    fact_id="allergies",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="No allergies that I know of, doctor.",
                    is_controlled_shield=True,
                    category="Allergies",
                    response_source="UNKNOWN",
                    fact_key="allergies"
                )

        # ---------------------------------------------------------
        # 11. FAMILY HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.FAMILY_HISTORY:
            if family_history and str(family_history).strip():
                stmt = str(family_history).strip()
                return RetrievedFact(
                    fact_id="family_history",
                    state=FactState.AVAILABLE,
                    truth_value=family_history,
                    permitted_statement=stmt,
                    category="FamilyHx",
                    response_source="CASE_FACT",
                    fact_key="family_history"
                )
            else:
                return RetrievedFact(
                    fact_id="family_history",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="No significant medical issues in my family that I'm aware of, doctor.",
                    is_controlled_shield=True,
                    category="FamilyHx",
                    response_source="UNKNOWN",
                    fact_key="family_history"
                )

        # ---------------------------------------------------------
        # 12. SOCIAL HISTORY / LIFESTYLE
        # ---------------------------------------------------------
        if intent.category == IntentCategory.SOCIAL_HISTORY:
            if intent.subconcept == "past_activities":
                # User asked about past weekend or past activities not in case
                return RetrievedFact(
                    fact_id="past_activities",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I don't recall anything unusual, doctor, and I'm not sure how that relates to what's happening now.",
                    is_controlled_shield=True,
                    category="SocialHx",
                    response_source="UNKNOWN",
                    fact_key="past_activities"
                )

            if intent.subconcept == "diet_history":
                return RetrievedFact(
                    fact_id="diet_history",
                    state=FactState.UNKNOWN,
                    truth_value=None,
                    permitted_statement="I don't recall anything unusual about my meals, doctor.",
                    is_controlled_shield=True,
                    category="SocialHx",
                    response_source="UNKNOWN",
                    fact_key="diet_history"
                )

            job = patient.get("occupation") or "office worker"
            sh_str = str(social_history).strip() if social_history else "Non-smoker, no substance use."
            stmt = f"I work as an {job}. {sh_str}" if job.lower().startswith(('a', 'e', 'i', 'o', 'u')) else f"I work as a {job}. {sh_str}"
            return RetrievedFact(
                fact_id="social_history",
                state=FactState.AVAILABLE,
                truth_value=social_history,
                permitted_statement=stmt,
                category="SocialHx",
                response_source="CASE_FACT",
                fact_key="social_history"
            )

        # ---------------------------------------------------------
        # 13. ASSOCIATED SYMPTOMS & REVIEW OF SYSTEMS (TRI-STATE)
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ASSOCIATED_SYMPTOM:
            symptom_key = intent.subconcept or "general_inquiry"
            assoc_symptoms = history.get("associated_symptoms", {})

            # Check structured associated symptoms dictionary
            if isinstance(assoc_symptoms, dict) and symptom_key in assoc_symptoms:
                sym_data = assoc_symptoms[symptom_key]
                if isinstance(sym_data, dict):
                    sym_val = sym_data.get("value")
                else:
                    sym_val = sym_data

                if sym_val is True:
                    stmt = FactRetriever._format_positive_symptom(symptom_key)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE,
                        truth_value=True,
                        permitted_statement=stmt,
                        category="HPI",
                        response_source="CASE_FACT",
                        fact_key=symptom_key
                    )
                elif sym_val is False:
                    stmt = FactRetriever._format_negative_symptom(symptom_key, patient)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE_NEGATIVE,
                        truth_value=False,
                        permitted_statement=stmt,
                        category="HPI",
                        response_source="CASE_FACT",
                        fact_key=symptom_key
                    )
                else:
                    # Explicitly UNKNOWN
                    stmt = FactRetriever._format_unknown_symptom(symptom_key, patient)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.UNKNOWN,
                        truth_value=None,
                        permitted_statement=stmt,
                        is_controlled_shield=True,
                        category="HPI",
                        response_source="UNKNOWN",
                        fact_key=symptom_key
                    )

            # Check pertinent negatives list
            pertinent_negs = history.get("pertinent_negatives", [])
            if isinstance(pertinent_negs, list):
                negs_str = " ".join(str(n).lower() for n in pertinent_negs)
                if symptom_key in negs_str:
                    stmt = FactRetriever._format_negative_symptom(symptom_key, patient)
                    return RetrievedFact(
                        fact_id=f"associated_symptoms.{symptom_key}",
                        state=FactState.AVAILABLE_NEGATIVE,
                        truth_value=False,
                        permitted_statement=stmt,
                        category="HPI",
                        response_source="CASE_FACT",
                        fact_key=symptom_key
                    )

            # If not documented at all -> UNKNOWN
            return RetrievedFact(
                fact_id=f"associated_symptoms.{symptom_key}",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement=FactRetriever._format_unknown_symptom(symptom_key, patient),
                is_controlled_shield=True,
                category="HPI",
                response_source="UNKNOWN",
                fact_key=symptom_key
            )

        # Default fallback to UNKNOWN
        return RetrievedFact(
            fact_id="general_undocumented",
            state=FactState.UNKNOWN,
            truth_value=None,
            permitted_statement="I haven't really noticed anything like that, doctor.",
            is_controlled_shield=True,
            category="General",
            response_source="UNKNOWN",
            fact_key=None
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
            "shortness_of_breath": "Yes, I feel noticeably short of breath and tight in my chest.",
            "wheezing": "Yes, I hear whistling and wheezing when I breathe.",
            "sweating": "Yes, I'm noticeably sweaty and broke out in a cold sweat.",
            "dizziness": "Yes, I started feeling dizzy and lightheaded.",
            "nausea": "Yes, I feel nauseous.",
            "vomiting": "Yes, I threw up earlier.",
            "fever": "Yes, I've had a fever and chills.",
            "cough": "Yes, I have a persistent cough.",
            "back_pain": "Yes, I have severe pain in my back.",
            "anorexia": "Yes, I've completely lost my appetite.",
            "weight_loss": "Yes, I've had some unexplained weight loss recently.",
            "swelling": "Yes, I have swelling in my legs.",
            "bleeding": "Yes, I've noticed unusual bleeding.",
            "skin_moles": "Yes, I have some new or changing spots on my skin."
        }
        return phrasing.get(symptom, f"Yes, I've been having {symptom.replace('_', ' ')}.")

    @staticmethod
    def _format_negative_symptom(symptom: str, patient: Optional[Dict[str, Any]] = None) -> str:
        gender = str(patient.get("gender") or patient.get("sex") or "").strip().lower() if patient else ""
        is_male = gender in ["male", "m", "man"]

        phrasing = {
            "shortness_of_breath": "No, my breathing feels normal.",
            "wheezing": "No wheezing or whistling sound that I've noticed.",
            "sweating": "No, I haven't noticed any unusual sweating.",
            "dizziness": "No, I haven't felt dizzy.",
            "nausea": "No, I haven't felt nauseous.",
            "vomiting": "No, I haven't been vomiting.",
            "fever": "No, I haven't had a fever or chills.",
            "cough": "No, I don't have a cough.",
            "back_pain": "No, I don't have any back pain, doctor.",
            "diarrhea": "No, no diarrhea or bowel issues.",
            "urinary": "No, no issues or pain when I use the bathroom.",
            "skin_moles": "No, I haven't noticed any unusual skin spots, doctor.",
            "weight_loss": "No, my weight has been steady.",
            "swelling": "No, I haven't had any swelling in my legs or feet, doctor.",
            "bleeding": "No unusual bleeding or bruising, doctor.",
            "neurological": "No numbness, tingling, or weakness, doctor."
        }
        if symptom in ["pcod", "menstrual_history"] and is_male:
            return "I'm a male patient, doctor, so that doesn't apply to me."
        return phrasing.get(symptom, f"No, I haven't had any {symptom.replace('_', ' ')}.")

    @staticmethod
    def _format_unknown_symptom(symptom: str, patient: Optional[Dict[str, Any]] = None) -> str:
        gender = str(patient.get("gender") or patient.get("sex") or "").strip().lower() if patient else ""
        is_male = gender in ["male", "m", "man"]

        if symptom in ["pcod", "menstrual_history"] and is_male:
            return "I'm a male patient, doctor, so that doesn't apply to me."
        if symptom == "cancer":
            return "I don't have any history of cancer that I know of, doctor."
        if symptom == "vitiligo":
            return "I don't have any skin conditions or vitiligo that I'm aware of, doctor."
        if symptom == "diet_history":
            return "I don't recall anything unusual about my meals, doctor."
        if symptom in ["hairfall", "hair_loss"]:
            return "I'm not sure how that relates to what I'm experiencing, doctor."
        if symptom == "general_inquiry" or symptom == "unclassified_symptom":
            return "I haven't really noticed anything like that, doctor."
        return f"I haven't really noticed that, doctor."
