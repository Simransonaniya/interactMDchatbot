"""
InteractMD — Advanced Medical Medication Understanding Test Suite.
Tests all 16 required verification dimensions:
1. Medication entity recognition (dosage, unit, frequency, form, route)
2. Generic drug recognition (amlodipine, atorvastatin, paracetamol, albuterol, etc.)
3. Brand recognition (Lipitor, Norvasc, Tylenol, Ventolin, Caduet, Nicip Plus)
4. Spelling correction & noisy clinician language (medcian, paracetomol, amlodipinee, atorvastin)
5. Medication history query resolution
6. Medication adherence (missed doses, regular taking)
7. Medication statement (clinician orders / directives)
8. Medication purpose queries
9. Medication dosage queries
10. Medication frequency queries
11. Medication route queries
12. Medication side effect queries
13. Medication allergy queries
14. Negation handling (negated medications must not become active meds)
15. Temporal context (today vs yesterday vs this morning)
16. Food vs medicine disambiguation (did you eat breakfast vs did you eat medcian)
17. Real Robert Chen clinical case grounding
"""

import pytest
import uuid

from medical_nlu.normalizer import MedicalNormalizer
from medical_nlu.medication_catalog import MedicationCatalog
from medical_nlu.terminology_service import TerminologyService
from medical_nlu.entity_extractor import MedicalEntityExtractor
from medical_nlu.intent_classifier import MedicalIntentClassifier, MedicalIntent
from question_classifier import QuestionClassifier, IntentCategory, ClassifiedIntent
from patient_state import PatientSimulationState, PatientStateManager, parse_raw_medication_string
from ai_orchestrator import ai_orchestrator


class Test01_MedicationEntityRecognition:
    def setup_method(self):
        self.extractor = MedicalEntityExtractor()

    def test_amlodipine_dosage_frequency(self):
        entities = self.extractor.extract_entities("Amlodipine 5 mg daily")
        meds = [e for e in entities if e.type == "MEDICATION"]
        doses = [e for e in entities if e.type == "DOSAGE"]
        freqs = [e for e in entities if e.type == "FREQUENCY"]
        
        assert len(meds) > 0
        assert meds[0].normalized == "amlodipine"
        assert len(doses) > 0
        assert doses[0].normalized == "5 mg"
        assert len(freqs) > 0
        assert freqs[0].normalized == "daily"

    def test_atorvastatin_nightly(self):
        entities = self.extractor.extract_entities("Atorvastatin 20 mg at night")
        meds = [e for e in entities if e.type == "MEDICATION"]
        doses = [e for e in entities if e.type == "DOSAGE"]
        freqs = [e for e in entities if e.type == "FREQUENCY"]

        assert len(meds) > 0
        assert meds[0].normalized == "atorvastatin"
        assert len(doses) > 0
        assert doses[0].normalized == "20 mg"
        assert len(freqs) > 0
        assert "night" in freqs[0].normalized

    def test_albuterol_inhaler_as_needed(self):
        entities = self.extractor.extract_entities("albuterol inhaler as needed")
        meds = [e for e in entities if e.type == "MEDICATION"]
        forms = [e for e in entities if e.type == "FORM"]
        freqs = [e for e in entities if e.type == "FREQUENCY"]

        assert len(meds) > 0
        assert meds[0].normalized == "albuterol"
        assert len(forms) > 0
        assert forms[0].normalized == "inhaler"
        assert len(freqs) > 0
        assert freqs[0].normalized == "as needed"


class Test02_GenericDrugRecognition:
    def setup_method(self):
        self.catalog = MedicationCatalog.get_instance()
        self.classifier = MedicalIntentClassifier()

    def test_generic_drugs(self):
        generics = ["amlodipine", "atorvastatin", "paracetamol", "sertraline", "escitalopram", "albuterol", "salbutamol", "aspirin", "lisinopril", "metoprolol"]
        for drug in generics:
            res = self.classifier.classify(f"are you on {drug}")
            assert res.intent == MedicalIntent.MEDICATION_HISTORY
            assert any(e.type == "MEDICATION" for e in res.entities)


class Test03_BrandRecognition:
    def setup_method(self):
        self.catalog = MedicationCatalog.get_instance()
        self.classifier = MedicalIntentClassifier()

    def test_brand_names(self):
        brands = ["norvasc", "lipitor", "tylenol", "ventolin", "zoloft", "lexapro", "ecosprin", "pan 40", "niciplus"]
        for brand in brands:
            res = self.classifier.classify(f"take {brand}")
            assert res.intent in [MedicalIntent.MEDICATION_STATEMENT, MedicalIntent.MEDICATION_NAME_FRAGMENT]


class Test04_SpellingCorrection:
    def test_spelling_variants(self):
        typos = [
            ("medcian", "medicine"),
            ("medicin", "medicine"),
            ("medecine", "medicine"),
            ("medcine", "medicine"),
            ("medecin", "medicine"),
            ("paracetomol", "paracetamol"),
            ("amlodipin", "amlodipine"),
            ("atorvastin", "atorvastatin"),
            ("sertrakine", "sertraline"),
        ]
        for typo, expected in typos:
            norm = MedicalNormalizer.normalize(f"did you take {typo}")
            assert expected in norm.normalized_text


class Test05_MedicationHistory:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_history_variations(self):
        queries = [
            "What medicines do you take?",
            "What medications are you currently on?",
            "What meds are you taking?",
            "What tablets do you take?",
            "Did you take any medicine?",
            "Did you eat medicine?",
            "Did you eat medcian?",
            "What medicine do you take every day?",
            "Which medications are you on?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_HISTORY
            assert res.topic == "medication"


class Test06_MedicationAdherence:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_adherence_queries(self):
        queries = [
            "Do you take your medicine regularly?",
            "Are you taking your medications?",
            "Did you miss any doses?",
            "Have you been skipping your tablets?",
            "Did you take your morning medicine?",
            "Do you forget to take your medication?",
            "Did you miss your atorvastatin?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_ADHERENCE


class Test07_MedicationStatement:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_medication_statements(self):
        statements = [
            "Take paracetamol.",
            "Take this tablet.",
            "You should take this medication.",
            "Take Nicip Plus.",
            "You can take the medicine.",
            "Start this tablet."
        ]
        for s in statements:
            res = self.classifier.classify(s)
            assert res.intent == MedicalIntent.MEDICATION_STATEMENT


class Test08_MedicationPurpose:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_purpose_queries(self):
        queries = [
            ("What is amlodipine for?", "amlodipine"),
            ("Why are you taking atorvastatin?", "atorvastatin"),
            ("What does this medicine treat?", "medicine"),
            ("What is this tablet for?", "tablet"),
        ]
        for q, target in queries:
            res = self.classifier.classify(q)
            assert res.intent in [MedicalIntent.MEDICATION_PURPOSE_QUERY, MedicalIntent.MEDICATION_EFFECT_QUERY]


class Test09_MedicationDosage:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_dosage_queries(self):
        queries = [
            "How much amlodipine do you take?",
            "What dose are you on?",
            "How many milligrams?",
            "What strength is the tablet?",
            "What dose of amlodipine do you take?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_DOSAGE_QUERY


class Test10_MedicationFrequency:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_frequency_queries(self):
        queries = [
            "How often do you take it?",
            "How many times a day?",
            "Do you take it every day?",
            "Morning or night?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_FREQUENCY_QUERY


class Test11_MedicationRoute:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_route_queries(self):
        queries = [
            "Is it a tablet?",
            "Is it an inhaler?",
            "Do you inject it?",
            "How do you take it?",
            "Is it oral?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_ROUTE_QUERY


class Test12_MedicationSideEffects:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_side_effect_queries(self):
        queries = [
            "Does this medicine cause dizziness?",
            "Any side effects?",
            "Did the tablet cause nausea?",
            "Are you having problems from the medication?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent == MedicalIntent.MEDICATION_SIDE_EFFECT_QUERY


class Test13_MedicationAllergies:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_allergy_queries(self):
        queries = [
            "Are you allergic to any medication?",
            "Any drug allergies?",
            "Are you allergic to penicillin?",
            "Which medicines are you allergic to?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            assert res.intent in [MedicalIntent.MEDICATION_ALLERGY_QUERY, MedicalIntent.ALLERGIES]


class Test14_NegationHandling:
    def setup_method(self):
        self.extractor = MedicalEntityExtractor()
        self.classifier = MedicalIntentClassifier()

    def test_negation_extraction(self):
        sentences = [
            "I don't take any medication.",
            "I stopped amlodipine.",
            "I am not taking atorvastatin.",
            "I never took that medicine.",
            "No, I don't take tablets."
        ]
        for s in sentences:
            res = self.classifier.classify(s)
            assert res.negated is True


class Test15_TemporalContext:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_temporal_differentiation(self):
        r_current = self.classifier.classify("What medicines do you normally take?")
        r_morning = self.classifier.classify("Did you take your medicine this morning?")
        
        assert r_current.intent == MedicalIntent.MEDICATION_HISTORY
        assert r_morning.intent == MedicalIntent.MEDICATION_ADHERENCE
        assert r_morning.temporal_reference in ["today", "this morning", "morning"]


class Test16_FoodVsMedicineDisambiguation:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_food_vs_meds_exact(self):
        assert self.classifier.classify("did you eat breakfast").intent == MedicalIntent.DIET_HISTORY
        assert self.classifier.classify("did you eat medicine").intent == MedicalIntent.MEDICATION_HISTORY
        assert self.classifier.classify("did you eat medcian").intent == MedicalIntent.MEDICATION_HISTORY
        assert self.classifier.classify("what did you eat for dinner").intent == MedicalIntent.DIET_HISTORY
        assert self.classifier.classify("what medicine did you take").intent == MedicalIntent.MEDICATION_HISTORY
        assert self.classifier.classify("take paracetamol").intent == MedicalIntent.MEDICATION_STATEMENT


class Test17_RealRobertChenCaseGrounding:
    def test_robert_chen_multi_turn_flow(self):
        session_id = f"test-rc-med-{uuid.uuid4()}"
        case_id = "chest_pain_001"

        # 1. "What medicines do you take?" -> Amlodipine 5 mg & Atorvastatin 20 mg
        r1 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="What medicines do you take?",
            session_id=session_id
        )
        rep1 = r1["reply"].lower()
        assert "amlodipine" in rep1
        assert "atorvastatin" in rep1

        # 2. "Did you eat medcian?" -> Medication history
        r2 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Did you eat medcian?",
            session_id=session_id
        )
        rep2 = r2["reply"].lower()
        assert "amlodipine" in rep2 or "atorvastatin" in rep2 or "medication" in rep2
        assert "what i ate" not in rep2

        # 3. "Did you eat breakfast?" -> Breakfast / Diet route
        r3 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Did you eat breakfast?",
            session_id=session_id
        )
        rep3 = r3["reply"].lower()
        assert "breakfast" in rep3 or "remember" in rep3
        assert "amlodipine" not in rep3

        # 4. "Did you miss your atorvastatin doses?" -> Adherence route
        r4 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Did you miss your atorvastatin doses?",
            session_id=session_id
        )
        rep4 = r4["reply"].lower()
        assert any(k in rep4 for k in ["miss", "forget", "busy"])

        # 5. "Take paracetamol." -> Medication statement route
        r5 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Take paracetamol.",
            session_id=session_id
        )
        rep5 = r5["reply"].lower()
        assert any(k in rep5 for k in ["okay", "doctor", "take", "help", "ease"])
        assert "my daily medications: paracetamol" not in rep5

        # 6. "What dose of amlodipine do you take?" -> Dose query
        r6 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="What dose of amlodipine do you take?",
            session_id=session_id
        )
        rep6 = r6["reply"].lower()
        assert "5" in rep6 and "mg" in rep6

        # 7. "How often do you take it?" -> Frequency query
        r7 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="How often do you take it?",
            session_id=session_id
        )
        rep7 = r7["reply"].lower()
        assert "daily" in rep7 or "once a day" in rep7 or "day" in rep7

        # 8. "What is amlodipine for?" -> Purpose query
        r8 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="What is amlodipine for?",
            session_id=session_id
        )
        rep8 = r8["reply"].lower()
        assert "blood pressure" in rep8 or "hypertension" in rep8

        # 9. "Are you allergic to any medicine?" -> Allergy query
        r9 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Are you allergic to any medicine?",
            session_id=session_id
        )
        rep9 = r9["reply"].lower()
        assert any(k in rep9 for k in ["no known", "nkda", "no", "allergies"])
