"""
InteractMD — Medical Language Understanding & Medication Disambiguation Test Suite.
Tests:
- Test 1: "did you eat medcian" -> MEDICATION_HISTORY, slot: current_medications
- Test 2: "did you eat medicine" -> MEDICATION_HISTORY
- Test 3: "did you take medicine" -> MEDICATION_HISTORY
- Test 4: "what medicine do you take" -> MEDICATION_HISTORY
- Test 5: "did you eat breakfast" -> DIET_HISTORY, slot: breakfast
- Test 6: "what did you eat for dinner" -> DIET_HISTORY, slot: dinner
- Test 7: "take paracetamol" -> MEDICATION_STATEMENT
- Test 8: "niciplus" -> MEDICATION_NAME_FRAGMENT
- Test 9: "paracetomol tablet" -> MEDICATION_STATEMENT / MEDICATION_NAME_FRAGMENT
- Test 10: "did you take your medicine today" -> MEDICATION_ADHERENCE
- Hard Negatives: Food vs Medicine disambiguation
- Context Continuity: Diet question followed by medication question with spelling error
- Full 10-turn Robert Chen clinical sequence
"""

import pytest
import uuid

from medical_nlu.normalizer import MedicalNormalizer
from medical_nlu.medication_lexicon import MedicationLexicon
from medical_nlu.entity_extractor import MedicalEntityExtractor
from medical_nlu.intent_classifier import MedicalIntentClassifier, MedicalIntent
from question_classifier import QuestionClassifier, IntentCategory, ClassifiedIntent
from ai_orchestrator import ai_orchestrator


class TestMedicalNormalization:
    def test_medication_spelling_corrections(self):
        norm1 = MedicalNormalizer.normalize("did you eat medcian")
        assert "medicine" in norm1.normalized_text

        norm2 = MedicalNormalizer.normalize("take paracetomol")
        assert "paracetamol" in norm2.normalized_text

        norm3 = MedicalNormalizer.normalize("take niciplus")
        assert "nicip plus" in norm3.normalized_text

        norm4 = MedicalNormalizer.normalize("had you breakfast")
        assert "breakfast" in norm4.normalized_text

        norm5 = MedicalNormalizer.normalize("are you taking amlodipin or atorvastin")
        assert "amlodipine" in norm5.normalized_text
        assert "atorvastatin" in norm5.normalized_text


class TestMedicalEntityExtraction:
    def test_entity_detection(self):
        extractor = MedicalEntityExtractor()
        
        entities = extractor.extract_entities("did you eat medcian")
        meds = [e for e in entities if e.type == "MEDICATION"]
        assert len(meds) > 0
        assert meds[0].normalized == "medicine"

        entities_food = extractor.extract_entities("did you eat breakfast")
        foods = [e for e in entities_food if e.type == "FOOD_MEAL"]
        assert len(foods) > 0
        assert foods[0].normalized == "breakfast"


class TestRegressionIntents:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_01_eat_medcian(self):
        res = self.classifier.classify("did you eat medcian")
        assert res.intent == MedicalIntent.MEDICATION_HISTORY
        assert res.slot == "current_medications"
        assert res.topic == "medication"
        assert any(e.type == "MEDICATION" for e in res.entities)

    def test_02_eat_medicine(self):
        res = self.classifier.classify("did you eat medicine")
        assert res.intent == MedicalIntent.MEDICATION_HISTORY
        assert res.slot == "current_medications"

    def test_03_take_medicine(self):
        res = self.classifier.classify("did you take medicine")
        assert res.intent == MedicalIntent.MEDICATION_HISTORY
        assert res.slot == "current_medications"

    def test_04_what_medicine_do_you_take(self):
        res = self.classifier.classify("what medicine do you take")
        assert res.intent == MedicalIntent.MEDICATION_HISTORY
        assert res.slot == "current_medications"

    def test_05_did_you_eat_breakfast(self):
        res = self.classifier.classify("did you eat breakfast")
        assert res.intent == MedicalIntent.DIET_HISTORY
        assert res.slot == "breakfast"

    def test_06_what_did_you_eat_for_dinner(self):
        res = self.classifier.classify("what did you eat for dinner")
        assert res.intent == MedicalIntent.DIET_HISTORY
        assert res.slot == "dinner"

    def test_07_take_paracetamol(self):
        res = self.classifier.classify("take paracetamol")
        assert res.intent == MedicalIntent.MEDICATION_STATEMENT
        assert res.treatment_substance == "paracetamol"

    def test_08_niciplus_fragment(self):
        res = self.classifier.classify("niciplus")
        assert res.intent == MedicalIntent.MEDICATION_NAME_FRAGMENT
        assert res.treatment_substance == "nicip plus"

    def test_09_paracetomol_tablet(self):
        res = self.classifier.classify("paracetomol tablet")
        assert res.intent in [MedicalIntent.MEDICATION_NAME_FRAGMENT, MedicalIntent.MEDICATION_STATEMENT]

    def test_10_did_you_take_your_medicine_today(self):
        res = self.classifier.classify("did you take your medicine today")
        assert res.intent == MedicalIntent.MEDICATION_ADHERENCE
        assert res.slot == "medication_adherence"


class TestHardNegativeFoodVsMedicine:
    def setup_method(self):
        self.classifier = MedicalIntentClassifier()

    def test_hard_negatives(self):
        # Pair 1: eat breakfast vs eat medicine
        r_food1 = self.classifier.classify("did you eat breakfast")
        r_med1 = self.classifier.classify("did you eat medicine")
        assert r_food1.intent == MedicalIntent.DIET_HISTORY
        assert r_med1.intent == MedicalIntent.MEDICATION_HISTORY

        # Pair 2: eat dinner vs eat medcian
        r_food2 = self.classifier.classify("what did you eat for dinner")
        r_med2 = self.classifier.classify("did you eat medcian")
        assert r_food2.intent == MedicalIntent.DIET_HISTORY
        assert r_med2.intent == MedicalIntent.MEDICATION_HISTORY

        # Pair 3: eat lunch vs take medicine
        r_food3 = self.classifier.classify("did you eat lunch")
        r_med3 = self.classifier.classify("did you take medicine")
        assert r_food3.intent == MedicalIntent.DIET_HISTORY
        assert r_med3.intent == MedicalIntent.MEDICATION_HISTORY


class TestFullRobertChenDialogueSequence:
    def test_robert_chen_production_flow(self):
        session_id = f"test-rc-prod-{uuid.uuid4()}"
        case_id = "chest_pain_001"

        # Turn 1: Initial complaint check / opening
        # Turn 2: "When did this start and how long has it lasted?" -> ONSET_TIMING
        r2 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="When did this start and how long has it lasted?",
            session_id=session_id
        )
        assert "45" in r2["reply"] or "minutes" in r2["reply"].lower()
        assert r2["intent"] == "HISTORY_QUESTION"

        # Turn 3: "Had you breakfast" -> DIET/BREAKFAST
        r3 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Had you breakfast",
            session_id=session_id
        )
        assert "breakfast" in r3["reply"].lower() or "remember" in r3["reply"].lower()
        assert "amlodipine" not in r3["reply"].lower()

        # Turn 4: "Have you experienced any cold sweats, nausea, or vomiting?" -> SWEATING/NAUSEA
        r4 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Have you experienced any cold sweats, nausea, or vomiting?",
            session_id=session_id
        )
        reply4 = r4["reply"].lower()
        assert any(k in reply4 for k in ["sweat", "sweating", "clammy", "nausea", "nauseous", "queasy"])

        # Turn 5: "did you eat medcian" -> MEDICATION HISTORY (Amlodipine / Atorvastatin)
        r5 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="did you eat medcian",
            session_id=session_id
        )
        reply5 = r5["reply"].lower()
        assert "i don't really remember what i ate" not in reply5
        assert "what i ate" not in reply5
        assert any(k in reply5 for k in ["amlodipine", "atorvastatin", "medication", "daily"])

        # Turn 6: "you should take tablet" -> MEDICATION STATEMENT
        r6 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="you should take tablet",
            session_id=session_id
        )
        assert any(k in r6["reply"].lower() for k in ["okay", "doctor", "take", "help", "ease"])

        # Turn 7: "Are you feeling short of breath or wheezing?" -> DYSPNEA
        r7 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="Are you feeling short of breath or wheezing?",
            session_id=session_id
        )
        assert any(k in r7["reply"].lower() for k in ["short of breath", "breath", "hard to catch", "yes"])

        # Turn 8: "can you tell me what type of medicine did you eat" -> MEDICATION HISTORY
        r8 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="can you tell me what type of medicine did you eat",
            session_id=session_id
        )
        reply8 = r8["reply"].lower()
        assert "what i ate" not in reply8
        assert any(k in reply8 for k in ["amlodipine", "atorvastatin", "medication", "daily"])

        # Turn 9: "what did you eat for dinner" -> DIET/DINNER
        r9 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="what did you eat for dinner",
            session_id=session_id
        )
        assert "dinner" in r9["reply"].lower() or "remember" in r9["reply"].lower()
        assert "amlodipine" not in r9["reply"].lower()

        # Turn 10: "did you take medicine" -> MEDICATION HISTORY
        r10 = ai_orchestrator.process_turn_sync(
            case_id=case_id,
            user_message="did you take medicine",
            session_id=session_id
        )
        assert any(k in r10["reply"].lower() for k in ["amlodipine", "atorvastatin", "medication", "prescribed", "daily"])
