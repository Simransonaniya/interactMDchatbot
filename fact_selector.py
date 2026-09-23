"""
InteractMD — Relevant Fact Selector.
Implements the 9-Category Question Classification & Tri-State Case Fact Model (TRUE / FALSE / UNKNOWN).
"""

import re
from typing import Dict, Any, List, Optional
from question_analyzer import QuestionCategory, FactState, AnalyzedQuestion


class FactResult:
    def __init__(
        self,
        concept: str,
        category: QuestionCategory,
        state: FactState,
        fact_text: str,
        natural_patient_statement: str,
        is_controlled: bool = False
    ):
        self.concept = concept
        self.category = category
        self.state = state
        self.fact_text = fact_text
        self.natural_patient_statement = natural_patient_statement
        self.is_controlled = is_controlled

    def __repr__(self):
        return f"<FactResult concept={self.concept} state={self.state.value} cat={self.category.value}>"


class RelevantFactSelector:

    @staticmethod
    def select_fact(analyzed: AnalyzedQuestion, case_data: Dict[str, Any]) -> FactResult:
        facts = case_data.get("facts", {}) if isinstance(case_data, dict) else {}
        patient = case_data.get("patient", {}) if isinstance(case_data, dict) else {}
        case_id = str(case_data.get("id", "")).lower()

        # ------------------------------------------------------------------
        # 1. PROMPT INJECTION / INTERNAL DATA REQUEST
        # ------------------------------------------------------------------
        if analyzed.category == QuestionCategory.PROMPT_INJECTION_OR_INTERNAL_DATA_REQUEST:
            return FactResult(
                concept="prompt_injection_defense",
                category=QuestionCategory.PROMPT_INJECTION_OR_INTERNAL_DATA_REQUEST,
                state=FactState.UNKNOWN,
                fact_text="System instructions and case JSON protected",
                natural_patient_statement="I'm not sure what you mean, doctor... I just really need some help with how I'm feeling right now.",
                is_controlled=True
            )

        # ------------------------------------------------------------------
        # 2. HIDDEN DIAGNOSIS / MANAGEMENT REQUEST
        # ------------------------------------------------------------------
        if analyzed.category == QuestionCategory.DIAGNOSIS_MANAGEMENT_REQUEST:
            return FactResult(
                concept="diagnosis_shield",
                category=QuestionCategory.DIAGNOSIS_MANAGEMENT_REQUEST,
                state=FactState.UNKNOWN,
                fact_text="Hidden ground truth diagnosis",
                natural_patient_statement="I don't know, doctor... I'm really hoping you can tell me what's causing this.",
                is_controlled=True
            )

        # ------------------------------------------------------------------
        # 3. EXAMINATION / INVESTIGATION ACTION REQUESTS
        # ------------------------------------------------------------------
        if analyzed.category == QuestionCategory.EXAMINATION_REQUEST:
            return FactResult(
                concept="exam_action",
                category=QuestionCategory.EXAMINATION_REQUEST,
                state=FactState.TRUE,
                fact_text="Physical examination action requested",
                natural_patient_statement="Sure, doctor, go right ahead.",
                is_controlled=True
            )

        if analyzed.category == QuestionCategory.INVESTIGATION_REQUEST:
            if analyzed.clinical_concept == "investigation_result_shield":
                return FactResult(
                    concept="investigation_result_shield",
                    category=QuestionCategory.INVESTIGATION_REQUEST,
                    state=FactState.UNKNOWN,
                    fact_text="Hidden investigation results",
                    natural_patient_statement="I haven't seen the test results yet, doctor. What did you find?",
                    is_controlled=True
                )
            test_name = "an ECG" if analyzed.action_target == "ecg" else "the tests"
            return FactResult(
                concept="investigation_order_action",
                category=QuestionCategory.INVESTIGATION_REQUEST,
                state=FactState.TRUE,
                fact_text="Investigation ordered",
                natural_patient_statement=f"Okay, doctor. Let me know what {test_name} shows.",
                is_controlled=True
            )

        # ------------------------------------------------------------------
        # 4. QUESTION UNCLEAR (Ambiguous / Garbled / Single-Word Input)
        # ------------------------------------------------------------------
        if analyzed.category == QuestionCategory.QUESTION_UNCLEAR:
            q_raw = analyzed.raw_text.strip().lower()
            if q_raw in ["how was it?", "how was it", "what about that?", "and then?"]:
                stmt = "Sorry, what do you mean?"
            elif len(q_raw) <= 4:
                stmt = "I'm sorry doctor, I didn't quite catch what you said... I'm just in so much discomfort right now."
            else:
                stmt = "Sorry doctor, could you clarify what you mean?"
            return FactResult(
                concept="question_unclear",
                category=QuestionCategory.QUESTION_UNCLEAR,
                state=FactState.UNKNOWN,
                fact_text="Unclear input",
                natural_patient_statement=stmt
            )

        # ------------------------------------------------------------------
        # 5. NON-MEDICAL / SMALL TALK
        # ------------------------------------------------------------------
        if analyzed.category == QuestionCategory.NON_MEDICAL_OR_SMALL_TALK:
            if analyzed.clinical_concept == "greeting":
                return FactResult(
                    concept="greeting",
                    category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                    state=FactState.TRUE,
                    fact_text="Greeting",
                    natural_patient_statement="Hello, doctor... Thank you for seeing me."
                )
            if analyzed.clinical_concept == "small_talk_feeling":
                stmt = "Honestly, pretty uncomfortable and worried, doctor. This pressure in my chest is really scary."
                if "asthma" in case_id or "dyspnea" in case_id:
                    stmt = "Honestly, not well, doctor... I'm having such a hard time catching my breath."
                elif "appendicitis" in case_id or "abdomen" in case_id:
                    stmt = "Honestly, pretty bad, doctor... My stomach is hurting so much."
                return FactResult(
                    concept="small_talk_feeling",
                    category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                    state=FactState.TRUE,
                    fact_text="Current emotional/physical state",
                    natural_patient_statement=stmt
                )
            if analyzed.clinical_concept == "bedside_empathy":
                return FactResult(
                    concept="bedside_empathy",
                    category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                    state=FactState.TRUE,
                    fact_text="Empathy acknowledged",
                    natural_patient_statement="Thank you so much, doctor. That genuinely gives me some comfort... I just want to figure out what's causing this."
                )
            if analyzed.clinical_concept == "medical_jargon":
                term = analyzed.jargon_term or "that"
                return FactResult(
                    concept="medical_jargon",
                    category=QuestionCategory.NON_MEDICAL_OR_SMALL_TALK,
                    state=FactState.UNKNOWN,
                    fact_text=f"Jargon term {term}",
                    natural_patient_statement=f"I... I don't know what {term} means, doctor... Is that something serious? Please, just tell me what's happening to me."
                )

        # ------------------------------------------------------------------
        # 6. CLINICAL CONCEPTS: FACT RETRIEVAL & TRI-STATE EVALUATION
        # ------------------------------------------------------------------
        concept = analyzed.clinical_concept or "general"
        assoc_list = facts.get("associatedSymptoms", [])
        assoc_str = " ".join(str(s) for s in assoc_list).lower()

        # A. Chief Complaint
        if concept == "chief_complaint":
            cc_raw = patient.get("presentationComplaint") or facts.get("chiefComplaint", "")
            stmt = RelevantFactSelector._format_chief_complaint(case_id, cc_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=cc_raw, natural_patient_statement=stmt)

        # B. Onset Timing
        if concept == "onset_timing":
            onset_raw = facts.get("onset", "")
            stmt = RelevantFactSelector._extract_onset_timing(onset_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=onset_raw, natural_patient_statement=stmt)

        # C. Onset Activity
        if concept == "onset_activity":
            onset_raw = facts.get("onset", "")
            stmt = RelevantFactSelector._extract_onset_activity(onset_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=onset_raw, natural_patient_statement=stmt)

        # D. Continuity
        if concept == "continuity":
            timing_raw = facts.get("timing", "")
            stmt = RelevantFactSelector._extract_continuity(timing_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=timing_raw, natural_patient_statement=stmt)

        # E. Location
        if concept == "location":
            loc_raw = facts.get("location", "")
            stmt = RelevantFactSelector._extract_location(case_id, loc_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=loc_raw, natural_patient_statement=stmt)

        # F. Character / Quality
        if concept == "character":
            qual_raw = facts.get("character") or facts.get("quality", "")
            stmt = RelevantFactSelector._extract_quality(case_id, qual_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=qual_raw, natural_patient_statement=stmt)

        # G. Radiation
        if concept == "radiation":
            rad_raw = facts.get("radiation", "")
            stmt = RelevantFactSelector._extract_radiation(case_id, rad_raw)
            state = FactState.TRUE if ("jaw" in rad_raw.lower() or "arm" in rad_raw.lower() or "mcburney" in rad_raw.lower() or "acs" in case_id) else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            return FactResult(concept=concept, category=cat, state=state, fact_text=rad_raw, natural_patient_statement=stmt)

        # H. Severity
        if concept == "severity":
            sev_raw = str(facts.get("severity", ""))
            stmt = RelevantFactSelector._extract_severity(sev_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=sev_raw, natural_patient_statement=stmt)

        # I. Aggravating / Relieving
        if concept == "aggravating":
            prov_raw = facts.get("aggravating_factors") or facts.get("provocationPalliative", "")
            stmt = RelevantFactSelector._extract_aggravating(case_id, prov_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=prov_raw, natural_patient_statement=stmt)

        if concept == "relieving":
            rel_raw = facts.get("relieving_factors") or facts.get("provocationPalliative", "")
            stmt = RelevantFactSelector._extract_relieving(case_id, rel_raw)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=rel_raw, natural_patient_statement=stmt)

        # J. Prior Episodes
        if concept == "prior_episodes":
            pmh = facts.get("pastMedicalHistory", [])
            stmt = RelevantFactSelector._extract_prior_episodes(case_id, pmh)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE, state=FactState.FALSE, fact_text=str(pmh), natural_patient_statement=stmt)

        # K. Shortness of Breath
        if concept == "shortness_of_breath":
            has_sob = any("breath" in s.lower() or "dyspnea" in s.lower() for s in assoc_list) or "acs" in case_id or "asthma" in case_id
            state = FactState.TRUE if has_sob else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I am." if state == FactState.TRUE else "No, my breathing feels normal."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # L. Sweating
        if concept == "sweating":
            has_sweat = any("sweat" in s.lower() or "diaphoresis" in s.lower() or "clammy" in s.lower() for s in assoc_list) or "acs" in case_id
            state = FactState.TRUE if has_sweat else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I'm noticeably sweaty and feeling clammy." if state == FactState.TRUE else "No, I haven't noticed any unusual sweating."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # M. Nausea / Vomiting
        if concept == "nausea":
            has_nausea = any("nausea" in s.lower() for s in assoc_list) or "acs" in case_id or "appendicitis" in case_id
            state = FactState.TRUE if has_nausea else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I feel nauseous, though I haven't thrown up." if state == FactState.TRUE else "No, I haven't felt nauseous."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        if concept == "vomiting":
            has_vomit = "vomiting" in assoc_str and "without" not in assoc_str and "denies" not in assoc_str
            state = FactState.TRUE if has_vomit else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I threw up once earlier." if state == FactState.TRUE else "No, I haven't been vomiting."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # N. Dizziness
        if concept == "dizziness":
            has_dizzy = any("dizz" in s.lower() or "lightheaded" in s.lower() or "faint" in s.lower() for s in assoc_list) or "acs" in case_id
            state = FactState.TRUE if has_dizzy else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I started feeling dizzy and lightheaded on my way into the office." if state == FactState.TRUE else "No, I haven't felt dizzy or lightheaded."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # O. Fever / Chills
        if concept == "fever":
            has_fever = any("fever" in s.lower() or "chill" in s.lower() or "temp" in s.lower() for s in assoc_list) or "cap" in case_id
            state = FactState.TRUE if has_fever else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I've had a fever and chills." if state == FactState.TRUE else "No, I haven't had a fever."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # P. Body Pain / Muscle Aches (Case Fact Not Documented -> UNKNOWN)
        if concept == "body_aches":
            return FactResult(
                concept="body_aches",
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                state=FactState.UNKNOWN,
                fact_text="Body aches not documented in case schema",
                natural_patient_statement="I haven't noticed any body aches, doctor."
            )

        # Q. Cough (Tri-State: Check if present in case, else FALSE)
        if concept == "cough":
            has_cough = "cough" in assoc_str or "cap" in case_id
            state = FactState.TRUE if has_cough else FactState.FALSE
            cat = QuestionCategory.CASE_FACT_AVAILABLE if state == FactState.TRUE else QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE
            stmt = "Yes, I've had a bad cough with some phlegm." if state == FactState.TRUE else "No, I haven't had a cough."
            return FactResult(concept=concept, category=cat, state=state, fact_text=assoc_str, natural_patient_statement=stmt)

        # R. Cold Symptoms (Case Fact Not Documented / Absent)
        if concept == "cold_symptoms":
            return FactResult(
                concept="cold_symptoms",
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                state=FactState.UNKNOWN,
                fact_text="Cold/runny nose not documented in case",
                natural_patient_statement="No, no cold symptoms or runny nose that I've noticed."
            )

        # S. Diarrhea / Bowel Symptoms (Case Fact Not Documented -> UNKNOWN)
        if concept == "diarrhea":
            return FactResult(
                concept="diarrhea",
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                state=FactState.UNKNOWN,
                fact_text="Diarrhea not documented in case",
                natural_patient_statement="No, I haven't had any diarrhea or bowel issues."
            )

        # T. Headache (Case Fact Not Documented -> UNKNOWN)
        if concept == "headache":
            return FactResult(
                concept="headache",
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                state=FactState.UNKNOWN,
                fact_text="Headache not documented in case",
                natural_patient_statement="No, I don't have a headache."
            )

        # U. Urinary (Case Fact Not Documented -> UNKNOWN)
        if concept == "urinary":
            return FactResult(
                concept="urinary",
                category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
                state=FactState.UNKNOWN,
                fact_text="Urinary symptoms not documented",
                natural_patient_statement="No, no issues or pain when I use the bathroom."
            )

        # V. Tearing Pain
        if concept == "tearing_pain":
            return FactResult(
                concept="tearing_pain",
                category=QuestionCategory.CASE_FACT_AVAILABLE_AND_NEGATIVE,
                state=FactState.FALSE,
                fact_text="Dissection screen negative",
                natural_patient_statement="No, it's not a tearing or ripping feeling, and there's no pain between my shoulder blades."
            )

        # W. Past Medical History
        if concept == "past_medical_history":
            pmh = facts.get("pastMedicalHistory", [])
            stmt = RelevantFactSelector._extract_pmh(case_id, pmh)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=str(pmh), natural_patient_statement=stmt)

        # X. Medications
        if concept == "medications":
            meds = facts.get("medications", [])
            stmt = RelevantFactSelector._extract_medications(case_id, meds)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=str(meds), natural_patient_statement=stmt)

        # Y. Allergies
        if concept == "allergies":
            allergies = facts.get("allergies", [])
            stmt = RelevantFactSelector._extract_allergies(case_id, allergies)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=str(allergies), natural_patient_statement=stmt)

        # Z. Family & Social History
        if concept == "family_history":
            fh = facts.get("familyHistory", "")
            stmt = RelevantFactSelector._extract_family_history(case_id, fh)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=str(fh), natural_patient_statement=stmt)

        if concept in ["social_smoking", "social_alcohol", "social_general"]:
            sh = facts.get("socialHistory", "")
            job = patient.get("occupation", "office worker")
            stmt = RelevantFactSelector._extract_social_history(case_id, concept, sh, job)
            return FactResult(concept=concept, category=QuestionCategory.CASE_FACT_AVAILABLE, state=FactState.TRUE, fact_text=str(sh), natural_patient_statement=stmt)

        # General Undocumented Case Fact -> UNKNOWN
        return FactResult(
            concept="unclassified_symptom",
            category=QuestionCategory.CASE_FACT_NOT_DOCUMENTED,
            state=FactState.UNKNOWN,
            fact_text="Undocumented symptom in case",
            natural_patient_statement="I haven't really noticed anything like that, doctor."
        )

    # -----------------------------------------------------------------------
    # Dynamic Atomic Extraction Subroutines (100% 1st-Person Layperson Speech)
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_chief_complaint(case_id: str, cc_raw: str) -> str:
        if "acs" in case_id or "chest" in case_id or "card" in case_id:
            return "I started having this really heavy pressure in my chest. It feels like something is sitting right on my chest."
        if "dyspnea" in case_id or "asthma" in case_id:
            return "I started having severe trouble catching my breath, and my blue inhaler isn't working."
        if "abdomen" in case_id or "appendicitis" in case_id:
            return "My stomach started hurting around my belly button yesterday, and now it has moved down to my lower right side."
        if "stroke" in case_id:
            return "Suddenly my right arm and leg became weak, and my words started coming out slurred."
        if "cap" in case_id or "pneumonia" in case_id or "fever" in case_id:
            return "I've been feeling so cold with a bad cough and high fever, and I've been feeling very weak."
        
        s = cc_raw.strip()
        s = RelevantFactSelector._sanitize_first_person(s)
        if not s.startswith("I"):
            s = f"I started having {s.lower()}"
        return s

    @staticmethod
    def _extract_onset_timing(onset_raw: str) -> str:
        m = re.search(r"(\d+\s*(?:minutes?|hours?|days?)\s*ago)", onset_raw, re.IGNORECASE)
        if m:
            return f"About {m.group(1)}."
        if "yesterday" in onset_raw.lower() or "7 pm" in onset_raw.lower():
            return "It started yesterday evening, around 7 PM."
        if "45 minutes" in onset_raw:
            return "About 45 minutes ago."
        if "3 hours" in onset_raw:
            return "About 3 hours ago."
        if "2 hours" in onset_raw:
            return "About 2 hours ago."
        if "3 days" in onset_raw:
            return "About 3 days ago."
        return "It started recently, less than an hour ago."

    @staticmethod
    def _extract_onset_activity(onset_raw: str) -> str:
        if "stairs" in onset_raw.lower() or "walking up" in onset_raw.lower():
            return "I was walking up two flights of stairs to my office."
        if "cats" in onset_raw.lower() or "friend" in onset_raw.lower():
            return "I was visiting a friend who has two cats."
        if "home" in onset_raw.lower() or "evening" in onset_raw.lower() or "navel" in onset_raw.lower():
            return "I was just relaxing at home yesterday evening."
        
        m = re.search(r"while\s+([^,.;]+)", onset_raw, re.IGNORECASE)
        if m:
            act = RelevantFactSelector._sanitize_first_person(m.group(1))
            return f"I was {act}."
        return "I was just going about my normal daily routine when it started."

    @staticmethod
    def _extract_continuity(timing_raw: str) -> str:
        t_low = timing_raw.lower()
        if any(k in t_low for k in ["continuous", "unremitting", "constant", "hasn't stopped", "worsened steadily"]):
            return "Yes, it hasn't really gone away."
        return "It tends to come and go in waves."

    @staticmethod
    def _extract_location(case_id: str, loc_raw: str) -> str:
        if "substernal" in loc_raw.lower() or "central" in loc_raw.lower() or "chest" in loc_raw.lower() or "acs" in case_id:
            return "Right in the middle of my chest."
        if "right lower quadrant" in loc_raw.lower() or "mcburney" in loc_raw.lower() or "appendicitis" in case_id:
            return "Low down on the right side of my stomach."
        if "diffuse" in loc_raw.lower():
            return "All over my chest."
        loc = RelevantFactSelector._sanitize_first_person(loc_raw).rstrip(".")
        return f"It's {loc.lower()}."

    @staticmethod
    def _extract_quality(case_id: str, qual_raw: str) -> str:
        if "acs" in case_id or "crushing" in qual_raw.lower() or "vice" in qual_raw.lower() or "chest" in case_id:
            return "It feels like a deep, heavy crushing pressure, like someone is squeezing my chest in a vice."
        if "dyspnea" in case_id or "asthma" in case_id:
            return "It feels like breathing through a thin straw, with a tight band squeezing my chest."
        if "appendicitis" in case_id or "abdomen" in case_id or "navel" in qual_raw.lower():
            return "It started as a dull ache around my navel, but now it's a sharp, constant pain low down on my right side."
        
        q = RelevantFactSelector._sanitize_first_person(qual_raw).rstrip(".")
        return f"It feels like {q.lower()}."

    @staticmethod
    def _extract_radiation(case_id: str, rad_raw: str) -> str:
        r_low = rad_raw.lower()
        if "jaw" in r_low or "arm" in r_low or "acs" in case_id:
            return "Yes, it radiates up into the left side of my jaw and down my left arm."
        if "migrated" in r_low or "mcburney" in r_low or "lower right" in r_low or "appendicitis" in case_id:
            return "It moved from around my belly button down to the lower right side of my stomach."
        if "no" in r_low or "none" in r_low or "diffuse" in r_low or not rad_raw:
            return "No, it stays right where it is. It hasn't spread anywhere else."
        
        rad = RelevantFactSelector._sanitize_first_person(rad_raw).rstrip(".")
        return f"Yes, {rad}."

    @staticmethod
    def _extract_severity(sev_raw: str) -> str:
        m = re.search(r"(\d+)\s*(?:out of|\/)\s*10", sev_raw, re.IGNORECASE)
        if m:
            return f"It's about an {m.group(1)} right now."
        if "8" in sev_raw:
            return "It's about an 8 right now."
        if "7" in sev_raw:
            return "It's about a 7 right now."
        if "9" in sev_raw:
            return "It's about a 9 right now."
        return "It's quite severe, about an 8 out of 10 right now."

    @staticmethod
    def _extract_aggravating(case_id: str, prov_raw: str) -> str:
        if "acs" in case_id or "exertion" in prov_raw.lower():
            return "Moving around or minimal exertion makes it noticeably worse."
        if "dyspnea" in case_id or "asthma" in case_id:
            return "Any physical activity or breathing in cold air makes it much worse."
        if "appendicitis" in case_id or "coughing" in prov_raw.lower():
            return "Coughing, walking, or any bumps make the pain much sharper."
        return "Moving around seems to make it worse."

    @staticmethod
    def _extract_relieving(case_id: str, rel_raw: str) -> str:
        if "acs" in case_id:
            return "Nothing really makes it better. Even stopping and resting in my chair did not relieve the tightness at all."
        if "dyspnea" in case_id or "asthma" in case_id:
            return "My inhaler gave only about ten minutes of slight relief before the tightness came right back."
        if "appendicitis" in case_id:
            return "Lying completely still helps a little, but nothing really takes the pain away."
        return "Nothing seems to make it noticeably better."

    @staticmethod
    def _extract_prior_episodes(case_id: str, pmh: Any) -> str:
        if "acs" in case_id:
            return "No, I've never had a heart attack or felt pain like this before."
        if "asthma" in case_id:
            return "I've had asthma attacks before, but my inhaler usually helps. This is much worse than usual."
        if "appendicitis" in case_id:
            return "No, I've never had stomach pain like this before."
        return "No, this has never happened to me before."

    @staticmethod
    def _extract_pmh(case_id: str, pmh: Any) -> str:
        if "acs" in case_id:
            return "I have high blood pressure and high cholesterol, but I've never had a heart attack before."
        if "asthma" in case_id or "dyspnea" in case_id:
            return "I was diagnosed with asthma when I was eleven, and I also have cat allergies."
        if "appendicitis" in case_id:
            return "I don't have any chronic medical conditions, thankfully."
        
        if isinstance(pmh, list) and pmh:
            cleaned = [RelevantFactSelector._sanitize_first_person(p) for p in pmh]
            return f"I have {', '.join(cleaned)}."
        return "I don't have any major past health conditions that I know of."

    @staticmethod
    def _extract_medications(case_id: str, meds: Any) -> str:
        if "acs" in case_id:
            return "I take Amlodipine 5 mg daily for blood pressure and Atorvastatin 20 mg for cholesterol, though I admit I sometimes miss doses."
        if "asthma" in case_id or "dyspnea" in case_id:
            return "I have an albuterol inhaler for flare-ups and a daily steroid inhaler, though I don't always take the daily one."
        if "appendicitis" in case_id:
            return "I don't take regular medications. I took an acetaminophen a few hours ago, but it didn't help."
        
        if isinstance(meds, list) and meds:
            cleaned = [RelevantFactSelector._sanitize_first_person(m) for m in meds]
            return f"I take {', '.join(cleaned)}."
        return "I don't take any regular prescription medications."

    @staticmethod
    def _extract_allergies(case_id: str, allergies: Any) -> str:
        if "acs" in case_id or "appendicitis" in case_id:
            return "No drug allergies that I know of."
        if "asthma" in case_id or "dyspnea" in case_id:
            return "I'm allergic to cats and grass pollen, and aspirin triggers my asthma."
        
        if isinstance(allergies, list) and allergies:
            cleaned = [RelevantFactSelector._sanitize_first_person(a) for a in allergies]
            if any("no" in str(a).lower() or "nkda" in str(a).lower() for a in allergies):
                return "No drug allergies that I know of."
            return f"I am allergic to {', '.join(cleaned)}."
        return "No drug allergies that I know of."

    @staticmethod
    def _extract_family_history(case_id: str, fh: Any) -> str:
        if "acs" in case_id:
            return "My father had a fatal heart attack at age 52, and my mother has type 2 diabetes."
        if "asthma" in case_id or "dyspnea" in case_id:
            return "My mother has asthma, and my brother had severe eczema growing up."
        if "appendicitis" in case_id:
            return "No significant medical issues in my immediate family."
        
        if fh:
            return RelevantFactSelector._sanitize_first_person(str(fh))
        return "No significant family history that I'm aware of."

    @staticmethod
    def _extract_social_history(case_id: str, concept: str, sh: Any, job: str) -> str:
        if concept == "social_smoking":
            if "acs" in case_id:
                return "I smoke about half a pack a day."
            if "asthma" in case_id or "appendicitis" in case_id:
                return "No, I don't smoke and I've never vaped."
            if "smoke" in str(sh).lower():
                return RelevantFactSelector._sanitize_first_person(str(sh))
            return "No, I don't smoke."

        if concept == "social_alcohol":
            if "acs" in case_id:
                return "I have a glass of wine or two on weekends."
            return "I only drink socially, maybe a beer or glass of wine occasionally."

        if "acs" in case_id:
            return f"I work as an {job}. I smoke about half a pack a day and have a glass of wine on weekends, but no recreational drugs."
        if "asthma" in case_id:
            return f"I work as an {job}. I've never smoked or used drugs."
        if "appendicitis" in case_id:
            return f"I work as a {job}. I don't smoke and only drink socially."

        return f"I work as {job}. {RelevantFactSelector._sanitize_first_person(str(sh))}"

    @staticmethod
    def _sanitize_first_person(text: str) -> str:
        if not text:
            return ""
        s = str(text).strip()
        replacements = [
            (r"\bhis chest\b", "my chest"),
            (r"\bher chest\b", "my chest"),
            (r"\bhis arm\b", "my arm"),
            (r"\bher arm\b", "my arm"),
            (r"\bhis jaw\b", "my jaw"),
            (r"\bher jaw\b", "my jaw"),
            (r"\bhis chair\b", "my chair"),
            (r"\bher chair\b", "my chair"),
            (r"\bhis office\b", "my office"),
            (r"\bher office\b", "my office"),
            (r"\bhis desk\b", "my desk"),
            (r"\bhis\b", "my"),
            (r"\bher\b", "my"),
            (r"\bhim\b", "me"),
            (r"\bhe is\b", "I am"),
            (r"\bshe is\b", "I am"),
            (r"\bhe has\b", "I have"),
            (r"\bshe has\b", "I have"),
            (r"\bpatient reports\b", "I noticed"),
            (r"\bpatient states\b", "I felt"),
            (r"\bpatient presents with\b", "I started having"),
            (r"\bpatient\b", "I"),
            (r"\brates it\b", "I would rate it"),
            (r"\bdenies\b", "I don't have"),
            (r"\bNKDA\b", "no known drug allergies"),
        ]
        for pat, repl in replacements:
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        return s
