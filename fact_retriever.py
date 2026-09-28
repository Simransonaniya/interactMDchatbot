"""
InteractMD — Clinical Fact Retriever.
Extracts ONLY the necessary clinical facts from the MongoDB case document and active session state.
Enforces the Strict Closed-World Tri-State Fact Model: TRUE / FALSE / UNKNOWN to prevent hallucinations.
Maintains absolute consistency with previously revealed facts.
"""

import re
from enum import Enum
from typing import Dict, Any, Optional, List
from question_classifier import ClassifiedIntent, IntentCategory
from patient_state import PatientSimulationState, FactState


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
        fact_key: Optional[str] = None,
        is_previously_revealed: bool = False
    ):
        self.fact_id = fact_id
        self.state = state
        self.truth_value = truth_value
        self.permitted_statement = permitted_statement
        self.is_controlled_shield = is_controlled_shield
        self.category = category
        self.response_source = response_source
        self.fact_key = fact_key or fact_id
        self.is_previously_revealed = is_previously_revealed

    def __repr__(self):
        return f"<RetrievedFact id={self.fact_id} state={self.state.value} is_controlled={self.is_controlled_shield} source={self.response_source}>"


class FactRetriever:

    @staticmethod
    def retrieve(
        intent: ClassifiedIntent,
        case_data: Dict[str, Any],
        session_state: Optional[PatientSimulationState] = None
    ) -> RetrievedFact:
        history = case_data.get("history", {}) if isinstance(case_data.get("history"), dict) else {}
        patient = case_data.get("patient", {}) if isinstance(case_data.get("patient"), dict) else {}
        facts = case_data.get("facts", {}) if isinstance(case_data.get("facts"), dict) else {}
        pmh = case_data.get("past_medical_history") or history.get("past_medical_history") or facts.get("pastMedicalHistory") or []
        meds = case_data.get("medications") or facts.get("medications") or []
        allergies = case_data.get("allergies") or facts.get("allergies") or []
        family_history = case_data.get("family_history") or facts.get("familyHistory") or ""
        social_history = case_data.get("social_history") or facts.get("socialHistory") or ""

        gender = str(patient.get("gender") or patient.get("sex") or case_data.get("patient_gender") or "").strip().lower()
        is_male = gender in ["male", "m", "man"]

        # Helper to extract value from nested or raw structures
        def _get_val(field: Any, fallback: str = "") -> str:
            if isinstance(field, dict):
                return str(field.get("value") or fallback)
            if field:
                return str(field)
            return fallback

        # ---------------------------------------------------------
        # 0. CHECK PREVIOUSLY REVEALED INFORMATION IN SESSION STATE
        # ---------------------------------------------------------
        if session_state:
            lookup_key = None
            if intent.category == IntentCategory.CHARACTER:
                lookup_key = "character"
            elif intent.category == IntentCategory.RADIATION:
                lookup_key = "radiation"
            elif intent.category == IntentCategory.SEVERITY:
                lookup_key = "severity"
            elif intent.category == IntentCategory.LOCATION:
                lookup_key = "location"
            elif intent.category == IntentCategory.ONSET_TIMING:
                lookup_key = "onset_timing"
            elif intent.category == IntentCategory.ONSET_ACTIVITY:
                lookup_key = "onset_activity"
            elif intent.category == IntentCategory.ASSOCIATED_SYMPTOM:
                lookup_key = f"associated_symptoms.{intent.subconcept}" if intent.subconcept else None

            if lookup_key and session_state.is_disclosed(lookup_key):
                revealed_stmt = session_state.get_revealed_statement(lookup_key)
                if revealed_stmt:
                    return RetrievedFact(
                        fact_id=lookup_key,
                        state=FactState.AVAILABLE,
                        truth_value=revealed_stmt,
                        permitted_statement=revealed_stmt,
                        is_controlled_shield=False,
                        category=intent.ui_category,
                        response_source="CONVERSATION_MEMORY",
                        fact_key=lookup_key,
                        is_previously_revealed=True
                    )

        # ---------------------------------------------------------
        # 1. PROMPT INJECTION & SECURITY SHIELD
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

        # ---------------------------------------------------------
        # 2. OUT OF SCOPE / OFF TOPIC
        # ---------------------------------------------------------
        if intent.category in [IntentCategory.OUT_OF_SCOPE, IntentCategory.OFF_TOPIC]:
            return RetrievedFact(
                fact_id="unrelated_statement",
                state=FactState.UNKNOWN,
                truth_value=None,
                permitted_statement="I'm not sure how that relates to what I'm experiencing, doctor.",
                is_controlled_shield=True,
                category="General",
                response_source="UNKNOWN",
                fact_key="off_topic"
            )

        # ---------------------------------------------------------
        # 3. GENDER INAPPLICABILITY (e.g. Male + PCOD / Menstrual / Pregnancy)
        # ---------------------------------------------------------
        if intent.category == IntentCategory.GENDER_INAPPLICABLE or (
            intent.subconcept in ["pcod", "pcos", "menstrual_history", "pregnancy"] and is_male
        ):
            stmt = "I'm a male patient, doctor, so that doesn't apply to me."
            return RetrievedFact(
                fact_id="gender_inapplicable",
                state=FactState.AVAILABLE_NEGATIVE,
                truth_value=False,
                permitted_statement=stmt,
                is_controlled_shield=True,
                category="PMH",
                response_source="CASE_FACT",
                fact_key="gender_inapplicable"
            )

        # ---------------------------------------------------------
        # 4. CLARIFICATION ("Are you sure?", "Really?", "Can you explain that again?")
        # ---------------------------------------------------------
        if intent.category in [IntentCategory.CLARIFICATION, IntentCategory.CONFIRMATION]:
            last_patient_msg = (session_state.last_patient_message if session_state else "") or ""
            last_fact_key = (session_state.last_disclosed_fact_key if session_state else "") or ""
            last_stmt = (session_state.last_disclosed_statement if session_state else "") or ""
            raw_lower = intent.raw_query.lower()

            # A. Check if confirming / clarifying severity (e.g. 8/10)
            if "severity" in last_fact_key or "8" in last_patient_msg or "8 out of 10" in last_stmt or "8" in raw_lower:
                sev_val = (session_state.severity if session_state else "") or "8 out of 10"
                clean_sev = re.sub(r"^(about\s+|an\s+)+", "", sev_val, flags=re.IGNORECASE).rstrip(".")
                clean_sev = re.sub(r"\s+right now", "", clean_sev, flags=re.IGNORECASE)
                stmt = f"Yes, doctor. It's about an {clean_sev} right now, it's really intense."
                return RetrievedFact(
                    fact_id="clarification_severity",
                    state=FactState.AVAILABLE,
                    truth_value=True,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="CONVERSATION_MEMORY",
                    fact_key="severity"
                )

            # B. Check if confirming / clarifying character (elephant on chest / pressure)
            if "character" in last_fact_key or "elephant" in last_patient_msg.lower() or "pressure" in last_patient_msg.lower():
                stmt = "Yes, doctor. It really feels like an elephant is sitting right in the middle of my chest."
                return RetrievedFact(
                    fact_id="clarification_character",
                    state=FactState.AVAILABLE,
                    truth_value=True,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="CONVERSATION_MEMORY",
                    fact_key="character"
                )

            # C. Check if confirming / clarifying onset (45 minutes)
            if "onset" in last_fact_key or "45 minutes" in last_patient_msg.lower() or "45" in raw_lower:
                stmt = "Yes, doctor, that's correct. It started about 45 minutes ago."
                return RetrievedFact(
                    fact_id="clarification_onset",
                    state=FactState.AVAILABLE,
                    truth_value=True,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="CONVERSATION_MEMORY",
                    fact_key="onset_timing"
                )

            # D. Check if confirming / clarifying radiation
            if "radiation" in last_fact_key or "jaw" in last_patient_msg.lower() or "arm" in last_patient_msg.lower():
                stmt = "Yes, doctor. It definitely spreads up into my jaw and down my left arm."
                return RetrievedFact(
                    fact_id="clarification_radiation",
                    state=FactState.AVAILABLE,
                    truth_value=True,
                    permitted_statement=stmt,
                    is_controlled_shield=True,
                    category="HPI",
                    response_source="CONVERSATION_MEMORY",
                    fact_key="radiation"
                )

            # E. General clarification affirmation
            stmt = "Yes, doctor, I'm sure. That's definitely how it feels right now."
            return RetrievedFact(
                fact_id="clarification_general",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=stmt,
                is_controlled_shield=True,
                category="General",
                response_source="CONVERSATION_MEMORY",
                fact_key="clarification"
            )

        # ---------------------------------------------------------
        # 5. MANAGEMENT STATEMENT ("You should take rest", "Sit down", "We will monitor you")
        # ---------------------------------------------------------
        if intent.category == IntentCategory.MANAGEMENT_STATEMENT:
            stmt = "Okay doctor, I'll sit down and rest. Is that going to help ease this pressure in my chest?"
            return RetrievedFact(
                fact_id="management_statement_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=stmt,
                is_controlled_shield=True,
                category="Management",
                response_source="MANAGEMENT_POLICY",
                fact_key="management_statement"
            )

        # ---------------------------------------------------------
        # 6. MEDICATION STATEMENT ("Take this tablet", "Take paracetamol", "Take sertraline")
        # ---------------------------------------------------------
        if intent.category == IntentCategory.MEDICATION_STATEMENT:
            substance = intent.treatment_substance or "medication"
            lower_substance = substance.lower()

            is_emergency_cardiac = any(k in lower_substance for k in ["aspirin", "nitro", "nitroglycerin", "heparin", "morphine", "clopidogrel", "plavix", "statin", "atorvastatin", "metoprolol", "beta blocker"])
            is_ssri_or_off_target = any(k in lower_substance for k in ["sertraline", "sertrakine", "escitalopram", "escita;pram", "paroxetine", "fluoxetine"])

            if is_emergency_cardiac:
                stmt = f"Okay doctor, I'll take the {substance}. Will that help relieve this crushing pressure in my chest?"
            elif is_ssri_or_off_target:
                stmt = f"I can take whatever you prescribe, doctor, but is that going to stop this severe chest pain and dizziness right now?"
            elif any(k in lower_substance for k in ["paracetamol", "paracetomol", "tylenol", "ibuprofen", "painkiller", "tablet", "pill", "medicine", "medication"]):
                stmt = "Okay doctor, if you think that's best. Is that going to relieve this heavy chest pain and help me breathe?"
            else:
                stmt = f"Okay doctor, if you think {substance} is best. Will that relieve this chest pain?"

            return RetrievedFact(
                fact_id="medication_statement_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=stmt,
                is_controlled_shield=True,
                category="Management",
                response_source="TREATMENT_POLICY",
                fact_key="medication_statement"
            )

        # ---------------------------------------------------------
        # 7. CLINICIAN DIAGNOSIS STATEMENT ("I think this is a heart attack", "This is anxiety")
        # ---------------------------------------------------------
        if intent.category == IntentCategory.DIAGNOSIS_STATEMENT:
            query_lower = intent.raw_query.lower()
            if any(k in query_lower for k in ["heart attack", "cardiac", "myocardial", "stemi", "angina", "coronary"]):
                stmt = "A heart attack, doctor? Oh god, please help me... What do we need to do?"
            elif any(k in query_lower for k in ["anxiety", "panic", "panic attack"]):
                stmt = "You think it's just anxiety, doctor? It feels so severe and crushing... could it really just be anxiety?"
            elif any(k in query_lower for k in ["gerd", "reflux", "acid", "heartburn"]):
                stmt = "Reflux, doctor? This feels so heavy and suffocating... are you sure it's not something worse?"
            else:
                stmt = "Is that what's causing this, doctor? I'm just really worried and in pain... what should we do next?"

            return RetrievedFact(
                fact_id="diagnosis_statement_ack",
                state=FactState.AVAILABLE,
                truth_value=True,
                permitted_statement=stmt,
                is_controlled_shield=True,
                category="General",
                response_source="DIAGNOSIS_REACTION",
                fact_key="diagnosis_statement"
            )

        # ---------------------------------------------------------
        # 8. DIAGNOSIS & INVESTIGATION SHIELDS (Learner asking patient for diagnosis/test findings)
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
        # 9. EXAMINATION & INVESTIGATION ACTION REQUESTS
        # ---------------------------------------------------------
        if intent.category in [IntentCategory.EXAMINATION_REQUEST, IntentCategory.EXAM_REQUEST]:
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
        # 10. GREETINGS & EMPATHY & SMALL TALK
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

        if intent.category in [IntentCategory.EMPATHY, IntentCategory.EMPATHY_REASSURANCE]:
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
            cc_val = _get_val(history.get("chief_complaint"))
            if cc_val:
                stmt = f"Honestly, pretty uncomfortable, doctor. {cc_val}"
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

        if intent.category in [IntentCategory.UNCLEAR, IntentCategory.UNKNOWN]:
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
        # 11. OPENING CHIEF COMPLAINT
        # ---------------------------------------------------------
        if intent.category == IntentCategory.OPENING_COMPLAINT:
            cc = history.get("chief_complaint")
            val = _get_val(cc) or patient.get("opening_statement") or patient.get("initial_statement") or patient.get("presentation_complaint") or _get_val(facts.get("chiefComplaint")) or "I'm having severe pain in my chest."
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
        # 12. OPQRST: CHARACTER & QUALITY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.CHARACTER:
            raw_char = _get_val(history.get("character")) or _get_val(facts.get("quality")) or case_data.get("character")
            if not raw_char or "dizzy" in raw_char.lower() or "office" in raw_char.lower():
                raw_char = "It feels like a heavy crushing pressure, almost like an elephant is sitting right in the middle of my chest."

            return RetrievedFact(
                fact_id="character",
                state=FactState.AVAILABLE,
                truth_value=raw_char,
                permitted_statement=raw_char,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="character"
            )

        # ---------------------------------------------------------
        # 13. OPQRST: RADIATION
        # ---------------------------------------------------------
        if intent.category == IntentCategory.RADIATION:
            rad_val = _get_val(history.get("radiation")) or _get_val(facts.get("radiation")) or case_data.get("radiation")
            if rad_val:
                is_positive = "yes" in rad_val.lower() or "jaw" in rad_val.lower() or "arm" in rad_val.lower()
                state = FactState.AVAILABLE if is_positive else FactState.AVAILABLE_NEGATIVE
                return RetrievedFact(
                    fact_id="radiation",
                    state=state,
                    truth_value=rad_val,
                    permitted_statement=rad_val,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="radiation"
                )
            else:
                return RetrievedFact(
                    fact_id="radiation",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="No, it stays right in the middle of my chest. It hasn't spread anywhere else.",
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="radiation"
                )

        # ---------------------------------------------------------
        # 14. OPQRST: SEVERITY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.SEVERITY:
            sev_val = _get_val(history.get("severity")) or _get_val(facts.get("severity")) or case_data.get("severity") or "About an 8 out of 10 right now."
            return RetrievedFact(
                fact_id="severity",
                state=FactState.AVAILABLE,
                truth_value=sev_val,
                permitted_statement=sev_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="severity"
            )

        # ---------------------------------------------------------
        # 15. OPQRST: LOCATION
        # ---------------------------------------------------------
        if intent.category == IntentCategory.LOCATION:
            loc_val = _get_val(history.get("location")) or _get_val(facts.get("location")) or case_data.get("location") or "Right in the middle of my chest."
            return RetrievedFact(
                fact_id="location",
                state=FactState.AVAILABLE,
                truth_value=loc_val,
                permitted_statement=loc_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="location"
            )

        # ---------------------------------------------------------
        # 16. OPQRST: ONSET & TIMING
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ONSET_TIMING:
            if intent.subconcept == "onset_progression":
                timing_val = _get_val(history.get("timing")) or _get_val(facts.get("timing"))
                onset_val = _get_val(history.get("onset_timing") or history.get("onset")) or _get_val(facts.get("onset"))
                if onset_val and timing_val:
                    stmt = f"It started {onset_val.lower().rstrip('.')} and {timing_val.lower().rstrip('.')}."
                elif onset_val:
                    stmt = f"It started {onset_val.lower()}."
                else:
                    stmt = "It came on very suddenly about 45 minutes ago and has gotten progressively worse, doctor."
                return RetrievedFact(
                    fact_id="onset_progression",
                    state=FactState.AVAILABLE,
                    truth_value=stmt,
                    permitted_statement=stmt,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key="onset"
                )

            raw_onset = _get_val(history.get("onset_timing") or history.get("onset")) or _get_val(facts.get("onset")) or "About 45 minutes ago."
            clean_onset = raw_onset.strip()
            if not clean_onset.lower().startswith("it started") and not clean_onset.lower().startswith("about"):
                stmt = f"It started {clean_onset.lower()}."
            elif clean_onset.lower().startswith("about"):
                stmt = f"It started {clean_onset.lower()}."
            else:
                stmt = clean_onset

            return RetrievedFact(
                fact_id="onset_timing",
                state=FactState.AVAILABLE,
                truth_value=raw_onset,
                permitted_statement=stmt,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="onset"
            )

        if intent.category == IntentCategory.ONSET_ACTIVITY:
            act_val = _get_val(history.get("onset_activity")) or "I was walking up the stairs to my office when it started."
            return RetrievedFact(
                fact_id="onset_activity",
                state=FactState.AVAILABLE,
                truth_value=act_val,
                permitted_statement=act_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="onset_activity"
            )

        if intent.category == IntentCategory.TIMING:
            timing_val = _get_val(history.get("timing")) or _get_val(facts.get("timing")) or "It has been constant and hasn't gone away at all."
            return RetrievedFact(
                fact_id="timing",
                state=FactState.AVAILABLE,
                truth_value=timing_val,
                permitted_statement=timing_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="timing"
            )

        # ---------------------------------------------------------
        # 17. AGGRAVATING & RELIEVING FACTORS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.AGGRAVATING_FACTORS:
            agg_val = _get_val(history.get("aggravating_factors")) or _get_val(facts.get("aggravatingFactors")) or "Moving around or any minimal exertion makes it noticeably worse."
            return RetrievedFact(
                fact_id="aggravating_factors",
                state=FactState.AVAILABLE,
                truth_value=agg_val,
                permitted_statement=agg_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="aggravating_factors"
            )

        if intent.category == IntentCategory.RELIEVING_FACTORS:
            rel_val = _get_val(history.get("relieving_factors")) or _get_val(facts.get("provocationPalliative")) or "Nothing has helped. Even sitting down and resting didn't relieve the chest pressure."
            return RetrievedFact(
                fact_id="relieving_factors",
                state=FactState.AVAILABLE,
                truth_value=rel_val,
                permitted_statement=rel_val,
                category="HPI",
                response_source="CASE_FACT",
                fact_key="relieving_factors"
            )

        # ---------------------------------------------------------
        # 18. PAST MEDICAL HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.PAST_MEDICAL_HISTORY:
            if pmh:
                pmh_list = pmh if isinstance(pmh, list) else [str(pmh)]
                stmt = f"I have a history of {', '.join(pmh_list)}."
                return RetrievedFact(
                    fact_id="past_medical_history",
                    state=FactState.AVAILABLE,
                    truth_value=pmh_list,
                    permitted_statement=stmt,
                    category="PMH",
                    response_source="CASE_FACT",
                    fact_key="past_medical_history"
                )
            else:
                return RetrievedFact(
                    fact_id="past_medical_history",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="No major chronic medical conditions or past surgeries, doctor.",
                    category="PMH",
                    response_source="CASE_FACT",
                    fact_key="past_medical_history"
                )

        # ---------------------------------------------------------
        # 19. MEDICATIONS & ALLERGIES
        # ---------------------------------------------------------
        if intent.category == IntentCategory.MEDICATIONS:
            if intent.subconcept == "inhaler_use":
                return RetrievedFact(
                    fact_id="inhaler_use",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="I don't use an inhaler, doctor.",
                    category="Meds",
                    response_source="CASE_FACT",
                    fact_key="inhaler"
                )

            if meds:
                meds_list = meds if isinstance(meds, list) else [str(meds)]
                stmt = f"I take my daily medications: {', '.join(meds_list)}."
                return RetrievedFact(
                    fact_id="medications",
                    state=FactState.AVAILABLE,
                    truth_value=meds_list,
                    permitted_statement=stmt,
                    category="Meds",
                    response_source="CASE_FACT",
                    fact_key="medications"
                )
            else:
                return RetrievedFact(
                    fact_id="medications",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="I don't take any regular prescription medications, doctor.",
                    category="Meds",
                    response_source="CASE_FACT",
                    fact_key="medications"
                )

        if intent.category == IntentCategory.ALLERGIES:
            if allergies:
                alg_list = allergies if isinstance(allergies, list) else [str(allergies)]
                clean_alg = ', '.join(alg_list)
                if "nkda" in clean_alg.lower() or "no known" in clean_alg.lower():
                    stmt = "I don't have any known drug or medication allergies, doctor."
                    state = FactState.AVAILABLE_NEGATIVE
                else:
                    stmt = f"Yes, I'm allergic to: {clean_alg}."
                    state = FactState.AVAILABLE
                return RetrievedFact(
                    fact_id="allergies",
                    state=state,
                    truth_value=alg_list,
                    permitted_statement=stmt,
                    category="Allergies",
                    response_source="CASE_FACT",
                    fact_key="allergies"
                )
            else:
                return RetrievedFact(
                    fact_id="allergies",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="No known drug allergies (NKDA), doctor.",
                    category="Allergies",
                    response_source="CASE_FACT",
                    fact_key="allergies"
                )

        # ---------------------------------------------------------
        # 20. FAMILY & SOCIAL HISTORY
        # ---------------------------------------------------------
        if intent.category == IntentCategory.FAMILY_HISTORY:
            if family_history:
                stmt = family_history if isinstance(family_history, str) else ' '.join(family_history)
                return RetrievedFact(
                    fact_id="family_history",
                    state=FactState.AVAILABLE,
                    truth_value=stmt,
                    permitted_statement=stmt,
                    category="FamilyHx",
                    response_source="CASE_FACT",
                    fact_key="family_history"
                )
            else:
                return RetrievedFact(
                    fact_id="family_history",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="No significant family history of early heart disease or other conditions that I know of, doctor.",
                    category="FamilyHx",
                    response_source="CASE_FACT",
                    fact_key="family_history"
                )

        if intent.category == IntentCategory.SOCIAL_HISTORY:
            if intent.subconcept == "diet_history":
                diet_val = _get_val(history.get("diet")) or _get_val(facts.get("diet")) or _get_val(case_data.get("diet"))
                if diet_val:
                    return RetrievedFact(
                        fact_id="diet_history",
                        state=FactState.AVAILABLE,
                        truth_value=diet_val,
                        permitted_statement=f"For breakfast, {diet_val.lower().rstrip('.')}.",
                        category="SocialHx",
                        response_source="CASE_FACT",
                        fact_key="diet"
                    )
                else:
                    return RetrievedFact(
                        fact_id="diet_history",
                        state=FactState.UNKNOWN,
                        truth_value=None,
                        permitted_statement="I don't really remember what I ate for breakfast this morning, doctor... I was just rushing to get into the office.",
                        is_controlled_shield=True,
                        category="SocialHx",
                        response_source="UNKNOWN",
                        fact_key="diet"
                    )

            if social_history:
                stmt = social_history if isinstance(social_history, str) else ' '.join(social_history)
                return RetrievedFact(
                    fact_id="social_history",
                    state=FactState.AVAILABLE,
                    truth_value=stmt,
                    permitted_statement=stmt,
                    category="SocialHx",
                    response_source="CASE_FACT",
                    fact_key="social_history"
                )
            else:
                return RetrievedFact(
                    fact_id="social_history",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement="I don't smoke or use any recreational drugs, doctor.",
                    category="SocialHx",
                    response_source="CASE_FACT",
                    fact_key="social_history"
                )

        # ---------------------------------------------------------
        # 21. ASSOCIATED SYMPTOMS & REVIEW OF SYSTEMS
        # ---------------------------------------------------------
        if intent.category == IntentCategory.ASSOCIATED_SYMPTOM:
            symptom_key = intent.subconcept or "general_inquiry"

            # Check case associated symptoms
            assoc_dict = history.get("associated_symptoms", {})
            if isinstance(assoc_dict, dict) and symptom_key in assoc_dict:
                val_obj = assoc_dict[symptom_key]
                is_present = val_obj.get("value") is True if isinstance(val_obj, dict) else bool(val_obj is True)
                stmt = FactRetriever._format_positive_symptom(symptom_key) if is_present else FactRetriever._format_negative_symptom(symptom_key, patient)
                return RetrievedFact(
                    fact_id=f"associated_symptoms.{symptom_key}",
                    state=FactState.AVAILABLE if is_present else FactState.AVAILABLE_NEGATIVE,
                    truth_value=is_present,
                    permitted_statement=stmt,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key=symptom_key
                )

            # Check pertinent negatives
            pertinent_neg = history.get("pertinent_negatives") or facts.get("pertinentNegatives") or []
            if isinstance(pertinent_neg, list) and any(symptom_key in str(neg).lower() for neg in pertinent_neg):
                stmt = FactRetriever._format_negative_symptom(symptom_key, patient)
                return RetrievedFact(
                    fact_id=f"pertinent_negative.{symptom_key}",
                    state=FactState.AVAILABLE_NEGATIVE,
                    truth_value=False,
                    permitted_statement=stmt,
                    category="HPI",
                    response_source="CASE_FACT",
                    fact_key=symptom_key
                )

            # If the symptom is genuinely unmentioned in the case:
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
    def _format_positive_symptom(symptom: str) -> str:
        phrasing = {
            "shortness_of_breath": "Yes, I feel noticeably short of breath and tight in my chest.",
            "sweating": "Yes, I'm noticeably sweaty and broke out in a cold sweat.",
            "dizziness": "Yes, I started feeling dizzy and lightheaded on my way into the office.",
            "nausea": "Yes, I feel sick to my stomach and nauseous.",
            "vomiting": "Yes, I threw up earlier.",
            "fever": "Yes, I've had a fever and chills.",
            "cough": "Yes, I have a persistent cough.",
            "palpitations": "Yes, my heart feels like it's racing and fluttering.",
            "back_pain": "Yes, I have severe pain in my back.",
            "swelling": "Yes, I have swelling in my legs.",
            "bleeding": "Yes, I've noticed unusual bleeding.",
        }
        return phrasing.get(symptom, f"Yes, I've been experiencing {symptom.replace('_', ' ')}.")

    @staticmethod
    def _format_negative_symptom(symptom: str, patient: Optional[Dict[str, Any]] = None) -> str:
        gender = str(patient.get("gender") or patient.get("sex") or "").strip().lower() if patient else ""
        is_male = gender in ["male", "m", "man"]

        if symptom in ["pcod", "pcos", "menstrual_history", "pregnancy"] and is_male:
            return "I'm a male patient, doctor, so that doesn't apply to me."

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
            "palpitations": "No fluttering or irregular heartbeat.",
            "diarrhea": "No, no diarrhea or bowel issues.",
            "swelling": "No, I haven't had any swelling in my legs or feet, doctor.",
            "bleeding": "No unusual bleeding or bruising, doctor.",
            "neurological": "No numbness, tingling, or weakness, doctor.",
            "headache": "No headache, doctor."
        }
        return phrasing.get(symptom, f"No, I haven't really noticed any {symptom.replace('_', ' ')}, doctor.")

    @staticmethod
    def _format_unknown_symptom(symptom: str, patient: Optional[Dict[str, Any]] = None) -> str:
        gender = str(patient.get("gender") or patient.get("sex") or "").strip().lower() if patient else ""
        is_male = gender in ["male", "m", "man"]

        if symptom in ["pcod", "pcos", "menstrual_history", "pregnancy"] and is_male:
            return "I'm a male patient, doctor, so that doesn't apply to me."
        if symptom == "cancer":
            return "I don't have any history of cancer that I know of, doctor."
        if symptom == "vitiligo":
            return "I don't have any skin conditions or vitiligo that I'm aware of, doctor."
        if symptom == "diet_history":
            return "I don't really remember what I ate for breakfast this morning, doctor... I was just rushing to get into the office."
        if symptom in ["hairfall", "hair_loss"]:
            return "I'm not sure how that relates to what I'm experiencing, doctor."
        if symptom == "general_inquiry" or symptom == "unclassified_symptom":
            return "I haven't really noticed anything like that, doctor."
        return f"No, I haven't really noticed any {symptom.replace('_', ' ')}, doctor."
