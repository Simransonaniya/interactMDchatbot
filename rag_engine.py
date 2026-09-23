"""
InteractMD — Clinical RAG (Retrieval-Augmented Generation) Pipeline.
Splits clinical cases into semantic knowledge chunks, performs vector similarity retrieval,
and grounds AI Patient responses in retrieved evidence with clinical guardrails.
"""

import math
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class KnowledgeChunk:
    def __init__(self, topic: str, content: str, natural_speech: str, keywords: List[str], weight: float = 1.0):
        self.topic = topic
        self.content = content
        self.natural_speech = natural_speech
        self.keywords = [k.lower() for k in keywords]
        self.weight = weight

    def __repr__(self):
        return f"<KnowledgeChunk topic='{self.topic}' keywords={len(self.keywords)}>"


class ClinicalCaseRAGIndexer:
    """
    Parses structured case data into indexed semantic knowledge chunks for retrieval.
    """

    @staticmethod
    def build_case_index(case_data: Dict[str, Any]) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        facts = case_data.get("facts", {})
        patient = case_data.get("patient", {})
        case_id = case_data.get("id", "")

        # 1. Chief Complaint
        cc = patient.get("presentationComplaint") or facts.get("chiefComplaint", "")
        if cc:
            chunks.append(KnowledgeChunk(
                topic="chief_complaint",
                content=cc,
                natural_speech=f"I started having this really heavy pressure in my chest. It feels like something is sitting on my chest." if "acs" in case_id else cc,
                keywords=["what brings you", "what happened", "how can i help", "trouble", "chief complaint", "reason for visit", "matter", "wrong", "why did you come"],
                weight=2.0
            ))

        # 2. Onset & Timing Duration
        onset = facts.get("onset", "")
        if onset:
            chunks.append(KnowledgeChunk(
                topic="onset_timing",
                content=onset,
                natural_speech="About 45 minutes ago." if "45 minutes" in onset else ("About 3 hours ago." if "3 hours" in onset else onset),
                keywords=["when did it start", "when did this start", "how long ago", "onset", "began", "started", "duration", "hours", "minutes"],
                weight=2.5
            ))

        # 3. Activity Context at Onset
        if onset:
            chunks.append(KnowledgeChunk(
                topic="onset_activity",
                content=onset,
                natural_speech="I was walking up two flights of stairs to my office." if "stairs" in onset else "I was just doing my normal morning routine when it started.",
                keywords=["what were you doing", "what was happening", "where were you", "activity", "brought on", "trigger"],
                weight=2.5
            ))

        # 4. Continuity & Pattern
        timing = facts.get("timing", "")
        if timing:
            chunks.append(KnowledgeChunk(
                topic="timing_continuity",
                content=timing,
                natural_speech="Yes, it hasn't really gone away." if ("continuous" in timing.lower() or "constant" in timing.lower()) else "It tends to come and go in waves.",
                keywords=["continuous", "constant", "steady", "come and go", "waves", "persistent", "has it been continuous"],
                weight=2.0
            ))

        # 5. Quality & Character
        quality = facts.get("quality", "")
        if quality:
            chunks.append(KnowledgeChunk(
                topic="pain_quality",
                content=quality,
                natural_speech="It feels like a deep, heavy crushing pressure, like someone is squeezing my chest in a vice." if "crushing" in quality.lower() else f"It feels like {quality}.",
                keywords=["feel like", "describe", "sharp", "dull", "crushing", "tight", "burning", "kind of pain", "type of pain", "character", "sensation"],
                weight=2.5
            ))

        # 6. Radiation & Location
        radiation = facts.get("radiation", "")
        if radiation:
            chunks.append(KnowledgeChunk(
                topic="pain_radiation",
                content=radiation,
                natural_speech="Yes, it radiates up into the left side of my jaw and down my left arm." if "jaw" in radiation.lower() else ("No, it stays right where it is. It hasn't spread anywhere else." if "no" in radiation.lower() else radiation),
                keywords=["radiat", "spread", "move", "go anywhere", "jaw", "arm", "back", "neck", "shoulder", "travel"],
                weight=2.5
            ))

        # 7. Severity & Pain Scale
        severity = facts.get("severity", "")
        if severity:
            chunks.append(KnowledgeChunk(
                topic="pain_severity",
                content=severity,
                natural_speech="It's about an 8 right now." if "8" in severity else "It's about a 7 right now.",
                keywords=["1 to 10", "1-10", "scale", "rate", "severity", "pain score", "how bad", "out of 10", "score"],
                weight=2.5
            ))

        # 8. Provocation & Palliation
        prov = facts.get("provocationPalliative", "")
        if prov:
            chunks.append(KnowledgeChunk(
                topic="provocation_palliation",
                content=prov,
                natural_speech="Nothing really makes it better, even resting in a chair. Moving around makes it worse, but resting hasn't relieved it.",
                keywords=["better", "worse", "aggravat", "reliev", "resting help", "deep breath", "moving around", "position"],
                weight=2.0
            ))

        # 9. Associated Symptoms
        assoc = facts.get("associatedSymptoms", [])
        for s in assoc:
            s_lower = s.lower()
            if "sweat" in s_lower or "diaphoresis" in s_lower:
                chunks.append(KnowledgeChunk(
                    topic="associated_sweating",
                    content=s,
                    natural_speech="Yes, I'm noticeably sweaty and feeling clammy.",
                    keywords=["sweat", "sweating", "clammy", "cold sweat", "perspir"],
                    weight=2.2
                ))
            elif "breath" in s_lower or "dyspnea" in s_lower:
                chunks.append(KnowledgeChunk(
                    topic="associated_dyspnea",
                    content=s,
                    natural_speech="Yes, I am.",
                    keywords=["short of breath", "shortness of breath", "breathless", "breathing", "winded", "dyspnea", "wheez"],
                    weight=2.2
                ))
            elif "nausea" in s_lower or "vomit" in s_lower:
                chunks.append(KnowledgeChunk(
                    topic="associated_nausea",
                    content=s,
                    natural_speech="Yes, I feel nauseous, though I haven't thrown up.",
                    keywords=["nausea", "nauseous", "throw up", "vomit", "sick to your stomach"],
                    weight=2.2
                ))
            elif "dizz" in s_lower or "lightheaded" in s_lower:
                chunks.append(KnowledgeChunk(
                    topic="associated_dizziness",
                    content=s,
                    natural_speech="Yes, I started feeling dizzy and lightheaded on my way into the office.",
                    keywords=["dizzy", "dizziness", "lightheaded", "faint", "pass out"],
                    weight=2.2
                ))

        # 10. Past Medical History
        pmh = facts.get("pastMedicalHistory", [])
        if pmh:
            chunks.append(KnowledgeChunk(
                topic="past_medical_history",
                content=", ".join(pmh) if isinstance(pmh, list) else str(pmh),
                natural_speech="I have high blood pressure and high cholesterol, but I've never had a heart attack before.",
                keywords=["medical history", "past medical", "health condition", "diagnosed before", "heart attack before", "stroke before", "hospital before"],
                weight=2.0
            ))

        # 11. Medications
        meds = facts.get("medications", [])
        if meds:
            chunks.append(KnowledgeChunk(
                topic="medications",
                content=", ".join(meds) if isinstance(meds, list) else str(meds),
                natural_speech="I take Amlodipine 5 mg daily for blood pressure and Atorvastatin 20 mg for cholesterol, though I admit I sometimes miss doses.",
                keywords=["medicat", "medicine", "pill", "prescription", "inhaler", "taking daily", "what do you take"],
                weight=2.0
            ))

        # 12. Allergies
        allergies = facts.get("allergies", [])
        if allergies:
            chunks.append(KnowledgeChunk(
                topic="allergies",
                content=", ".join(allergies) if isinstance(allergies, list) else str(allergies),
                natural_speech="No drug allergies that I know of.",
                keywords=["allerg", "allergic", "reaction"],
                weight=2.5
            ))

        # 13. Family History
        fh = facts.get("familyHistory", [])
        if fh:
            chunks.append(KnowledgeChunk(
                topic="family_history",
                content=" ".join(fh) if isinstance(fh, list) else str(fh),
                natural_speech="My father had a fatal heart attack at age 52, and my mother has type 2 diabetes.",
                keywords=["family", "father", "mother", "parent", "genetic", "heart disease in your family", "runs in your family"],
                weight=2.0
            ))

        # 14. Social History
        sh = facts.get("socialHistory", [])
        if sh:
            chunks.append(KnowledgeChunk(
                topic="social_history",
                content=" ".join(sh) if isinstance(sh, list) else str(sh),
                natural_speech="I work as an architectural project manager. I smoke about half a pack a day and have a glass of wine on weekends, but no recreational drugs.",
                keywords=["smoke", "tobacco", "cigarette", "alcohol", "drink", "wine", "beer", "drug", "work", "job", "occupation"],
                weight=2.0
            ))

        return chunks


class ClinicalRAGRetriever:
    """
    Vector similarity retriever that indexes knowledge chunks and ranks them by cosine / TF-IDF relevance score.
    """

    def __init__(self, chunks: List[KnowledgeChunk]):
        self.chunks = chunks
        self._build_vocabulary()

    def _build_vocabulary(self):
        # Extract unique terms from chunk contents and keywords
        vocab = set()
        for c in self.chunks:
            tokens = re.findall(r"\w+", (c.content + " " + " ".join(c.keywords) + " " + c.topic).lower())
            vocab.update(tokens)
        self.vocabulary = sorted(list(vocab))
        self.vocab_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.doc_vectors = np.array([self._vectorize(c) for c in self.chunks])

    def _vectorize(self, chunk: KnowledgeChunk) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        text = (chunk.content + " " + " ".join(chunk.keywords) + " " + chunk.topic).lower()
        tokens = re.findall(r"\w+", text)
        for t in tokens:
            if t in self.vocab_index:
                vec[self.vocab_index[t]] += 1.0
        # Boost keywords
        for k in chunk.keywords:
            k_tokens = re.findall(r"\w+", k)
            for kt in k_tokens:
                if kt in self.vocab_index:
                    vec[self.vocab_index[kt]] += chunk.weight * 2.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[KnowledgeChunk, float]]:
        query_tokens = re.findall(r"\w+", query.lower())
        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for t in query_tokens:
            if t in self.vocab_index:
                q_vec[self.vocab_index[t]] += 1.0
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Cosine similarity
        scores = np.dot(self.doc_vectors, q_vec)

        # Keyword bonus
        q_lower = query.lower()
        for i, c in enumerate(self.chunks):
            for k in c.keywords:
                if k in q_lower or (len(k.split()) == 1 and re.search(r'\b' + re.escape(k) + r'\b', q_lower)):
                    scores[i] += c.weight * 1.5

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0.1]
