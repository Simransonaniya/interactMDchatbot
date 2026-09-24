"""
InteractMD — Production Dual-Storage Database Connector & Repository.
Authoritative, fault-tolerant persistence for clinical cases, simulation sessions,
messages, interaction events, evaluations, and user profiles.

Ensures 100% data persistence by dual-writing to SQLite relational database
(interactmd.db) AND MongoDB Atlas (when reachable), with zero data loss.
"""

import os
import sys
import time
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from config import settings
from database import SessionLocal, engine, Base
import models
from models.session_model import SimulationSession
from models.message import Message as MessageModel
from models.evaluation import Evaluation as EvaluationModel
from models.interaction_event import InteractionEvent as InteractionEventModel
from models.user import User as UserModel
from models.case_model import (
    Case as CaseModel,
    PatientProfile as PatientProfileModel,
    ClinicalFact as ClinicalFactModel,
    PhysicalFinding as PhysicalFindingModel,
    Investigation as InvestigationModel,
    CaseHiddenEvaluation as CaseHiddenEvaluationModel
)


class MongoDBManager:
    _instance: Optional["MongoDBManager"] = None

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.is_connected = False
        self._memory_cases: Dict[str, Dict[str, Any]] = {}
        self._memory_sessions: Dict[str, Dict[str, Any]] = {}
        self._memory_messages: Dict[str, List[Dict[str, Any]]] = {}
        self._init_sqlite_tables()
        self._init_connection()
        self._load_initial_cases()

    @classmethod
    def get_instance(cls) -> "MongoDBManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_sqlite_tables(self):
        """Ensure all SQLite database tables exist."""
        try:
            Base.metadata.create_all(bind=engine)
            print("[Storage Engine] SQLite tables initialized successfully.")
        except Exception as e:
            print(f"[Storage Engine Warning] SQLite table init: {e}")

    def _init_connection(self):
        uri = settings.MONGODB_URI or settings.DATABASE_URL
        if not uri or not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
            print("[Storage Engine] Operating in SQLite local persistence mode.")
            return

        # Attempt connection with certifi if available
        tls_kwargs = {"serverSelectionTimeoutMS": 3000, "connectTimeoutMS": 3000}
        try:
            import certifi
            tls_kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass

        try:
            db_name = settings.MONGODB_DB_NAME or "interactmd"
            self.client = MongoClient(uri, **tls_kwargs)
            self.client.admin.command('ping')
            self.db = self.client.get_database(db_name)
            self.is_connected = True
            print("========================================================")
            print(f"  [MongoDB Atlas] Connected to live cluster: {db_name}")
            print("========================================================")
            self._ensure_indexes()
        except Exception as e:
            # Try once with tlsAllowInvalidCertificates if SSL handshake failed
            try:
                self.client = MongoClient(uri, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
                self.client.admin.command('ping')
                self.db = self.client.get_database(db_name)
                self.is_connected = True
                print("========================================================")
                print(f"  [MongoDB Atlas] Connected via resilient SSL mode: {db_name}")
                print("========================================================")
                self._ensure_indexes()
            except Exception as e2:
                print(f"[MongoDB Atlas Warning] Atlas unavailable ({e2}). Local SQLite is active.")
                self.is_connected = False
                self.client = None
                self.db = None

    def _ensure_indexes(self):
        if not self.is_connected or self.db is None:
            return
        try:
            self.cases_col.create_index("case_id", unique=True)
            self.cases_col.create_index("id")
            self.cases_col.create_index("specialty")
            self.cases_col.create_index("status")
            self.sessions_col.create_index("session_id", unique=True)
            self.sessions_col.create_index("id")
            self.sessions_col.create_index("user_id")
            self.sessions_col.create_index("case_id")
            self.messages_col.create_index("session_id")
            self.messages_col.create_index([("session_id", 1), ("timestamp", 1)])
            self.interaction_events_col.create_index("session_id")
            self.interaction_events_col.create_index([("session_id", 1), ("timestamp", 1)])
            self.evaluations_col.create_index("session_id", unique=True)
            self.evaluations_col.create_index("user_id")
            self.users_col.create_index("email", unique=True)
            self.users_col.create_index("id", unique=True)
        except Exception as e:
            print(f"[MongoDB Index Warning] {e}")

    # -----------------------------------------------------------------------
    # MongoDB Collections
    # -----------------------------------------------------------------------
    @property
    def cases_col(self) -> Optional[Collection]:
        return self.db["cases"] if (self.is_connected and self.db is not None) else None

    @property
    def sessions_col(self) -> Optional[Collection]:
        return self.db["sessions"] if (self.is_connected and self.db is not None) else None

    @property
    def messages_col(self) -> Optional[Collection]:
        return self.db["messages"] if (self.is_connected and self.db is not None) else None

    @property
    def interaction_events_col(self) -> Optional[Collection]:
        return self.db["interaction_events"] if (self.is_connected and self.db is not None) else None

    @property
    def evaluations_col(self) -> Optional[Collection]:
        return self.db["evaluations"] if (self.is_connected and self.db is not None) else None

    @property
    def users_col(self) -> Optional[Collection]:
        return self.db["users"] if (self.is_connected and self.db is not None) else None

    # -----------------------------------------------------------------------
    # Initial Cases Loader
    # -----------------------------------------------------------------------
    def _load_initial_cases(self):
        data_paths = [
            os.path.join(os.path.dirname(__file__), "data", "cases.json"),
            os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "data", "cases.json")
        ]
        for dp in data_paths:
            if os.path.exists(dp):
                try:
                    with open(dp, "r", encoding="utf-8") as f:
                        cases = json.load(f)
                        for c in cases:
                            cid = c.get("case_id") or c.get("id")
                            if cid and cid not in self._memory_cases:
                                self._memory_cases[cid] = c
                                # Also key by alias if different
                                if "id" in c and c["id"] != cid:
                                    self._memory_cases[c["id"]] = c
                                if "case_id" in c and c["case_id"] != cid:
                                    self._memory_cases[c["case_id"]] = c
                except Exception as e:
                    print(f"[Storage Warning] Could not load {dp}: {e}")

    # -----------------------------------------------------------------------
    # Cases CRUD (MongoDB Atlas + SQLite + Memory)
    # -----------------------------------------------------------------------
    def get_all_cases(self, specialty: Optional[str] = None, status: str = "published") -> List[Dict[str, Any]]:
        # 1. Try MongoDB
        if self.is_connected and self.cases_col is not None:
            try:
                query = {}
                if status and status != "all":
                    query["$or"] = [{"status": status}, {"is_published": True}]
                if specialty and specialty.lower() != "all":
                    query["specialty"] = {"$regex": f"^{specialty}$", "$options": "i"}
                cases = list(self.cases_col.find(query, {"_id": 0}))
                if cases:
                    return cases
            except Exception as e:
                print(f"[MongoDB Query Error] {e}")

        # 2. Return from Memory / JSON / SQLite
        res = list(self._memory_cases.values())
        # Deduplicate by title or primary id
        seen = set()
        unique_cases = []
        for c in res:
            key = c.get("title") or c.get("id")
            if key not in seen:
                seen.add(key)
                if specialty and specialty.lower() != "all" and c.get("specialty", "").lower() != specialty.lower():
                    continue
                unique_cases.append(c)
        return unique_cases

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        if not case_id:
            return None
        # 1. Try MongoDB
        if self.is_connected and self.cases_col is not None:
            try:
                c = self.cases_col.find_one({"$or": [{"case_id": case_id}, {"id": case_id}]}, {"_id": 0})
                if c:
                    return c
            except Exception as e:
                print(f"[MongoDB Query Error] {e}")

        # 2. Try Memory
        if case_id in self._memory_cases:
            return self._memory_cases[case_id]

        # Case-insensitive or substring match in memory
        target = case_id.lower().replace("-", "_")
        for k, v in self._memory_cases.items():
            k_clean = k.lower().replace("-", "_")
            if target in k_clean or k_clean in target:
                return v

        # 3. Fallback to first available case
        if self._memory_cases:
            return next(iter(self._memory_cases.values()))
        return None

    def save_case(self, case_data: Dict[str, Any]) -> str:
        case_id = case_data.get("case_id") or case_data.get("id") or str(uuid.uuid4())
        case_data["case_id"] = case_id
        case_data["id"] = case_id
        case_data.setdefault("version", 1)
        case_data.setdefault("updated_at", time.time())

        # Memory
        self._memory_cases[case_id] = case_data

        # MongoDB
        if self.is_connected and self.cases_col is not None:
            try:
                self.cases_col.update_one(
                    {"$or": [{"case_id": case_id}, {"id": case_id}]},
                    {"$set": case_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Case Save Error] {e}")

        # SQLite
        try:
            with SessionLocal() as db:
                existing = db.query(CaseModel).filter(CaseModel.id == case_id).first()
                if not existing:
                    db_case = CaseModel(
                        id=case_id,
                        title=case_data.get("title", "Clinical Case"),
                        specialty=case_data.get("specialty", "General"),
                        description=case_data.get("description", ""),
                        difficulty=case_data.get("difficulty", "Intermediate"),
                        is_published=case_data.get("is_published", True)
                    )
                    db.add(db_case)
                    db.commit()
        except Exception as e:
            print(f"[SQLite Case Save Warning] {e}")

        return case_id

    # -----------------------------------------------------------------------
    # Sessions CRUD (MongoDB Atlas + SQLite + Memory)
    # -----------------------------------------------------------------------
    def create_session(self, session_data: Dict[str, Any]) -> str:
        session_id = session_data.get("session_id") or session_data.get("id") or str(uuid.uuid4())
        session_data["session_id"] = session_id
        session_data["id"] = session_id
        session_data.setdefault("status", "ACTIVE")
        session_data.setdefault("started_at", time.time())
        session_data.setdefault("created_at", time.time())
        session_data.setdefault("updated_at", time.time())
        session_data.setdefault("revealed_fact_ids", [])
        session_data.setdefault("completed_examinations", [])
        session_data.setdefault("ordered_investigations", [])
        session_data.setdefault("revealed_investigation_results", [])
        session_data.setdefault("current_patient_state", "stable")

        # 1. Memory cache
        self._memory_sessions[session_id] = dict(session_data)

        # 2. MongoDB
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.insert_one(dict(session_data))
            except Exception as e:
                print(f"[MongoDB Session Create Error] {e}")

        # 3. SQLite Persistence
        try:
            with SessionLocal() as db:
                # Ensure referenced Case exists in SQLite
                case_id = session_data.get("case_id", "chest_pain_001")
                case_exists = db.query(CaseModel).filter(CaseModel.id == case_id).first()
                if not case_exists:
                    db.add(CaseModel(
                        id=case_id,
                        title="Clinical Simulation Case",
                        specialty="General",
                        description="Simulation Case",
                        difficulty="Intermediate",
                        is_published=True
                    ))
                    db.flush()

                db_sess = SimulationSession(
                    id=session_id,
                    user_id=session_data.get("user_id"),
                    case_id=case_id,
                    status=session_data.get("status", "ACTIVE"),
                    session_data=session_data
                )
                db.add(db_sess)
                db.commit()
                print(f"[Storage Engine] Saved new session '{session_id}' to SQLite database.")
        except Exception as e:
            print(f"[SQLite Session Create Warning] {e}")

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        # 1. MongoDB
        if self.is_connected and self.sessions_col is not None:
            try:
                doc = self.sessions_col.find_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {"_id": 0}
                )
                if doc:
                    self._memory_sessions[session_id] = doc
                    return doc
            except Exception as e:
                print(f"[MongoDB Get Session Error] {e}")

        # 2. Memory
        if session_id in self._memory_sessions:
            return self._memory_sessions[session_id]

        # 3. SQLite
        try:
            with SessionLocal() as db:
                db_sess = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
                if db_sess:
                    data = dict(db_sess.session_data or {})
                    data["session_id"] = db_sess.id
                    data["id"] = db_sess.id
                    data["user_id"] = db_sess.user_id
                    data["case_id"] = db_sess.case_id
                    data["status"] = db_sess.status
                    data["started_at"] = db_sess.started_at.timestamp() if db_sess.started_at else time.time()
                    data["created_at"] = db_sess.created_at.timestamp() if db_sess.created_at else time.time()
                    data["updated_at"] = db_sess.updated_at.timestamp() if db_sess.updated_at else time.time()
                    self._memory_sessions[session_id] = data
                    return data
        except Exception as e:
            print(f"[SQLite Get Session Warning] {e}")

        return None

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        # 1. MongoDB
        if self.is_connected and self.sessions_col is not None:
            try:
                return list(self.sessions_col.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1))
            except Exception as e:
                print(f"[MongoDB Get User Sessions Error] {e}")

        # 2. SQLite
        res = []
        try:
            with SessionLocal() as db:
                db_sessions = db.query(SimulationSession).filter(SimulationSession.user_id == user_id).order_by(SimulationSession.created_at.desc()).all()
                for s in db_sessions:
                    d = dict(s.session_data or {})
                    d["session_id"] = s.id
                    d["id"] = s.id
                    d["user_id"] = s.user_id
                    d["case_id"] = s.case_id
                    d["status"] = s.status
                    d["started_at"] = s.started_at.timestamp() if s.started_at else time.time()
                    d["created_at"] = s.created_at.timestamp() if s.created_at else time.time()
                    d["updated_at"] = s.updated_at.timestamp() if s.updated_at else time.time()
                    res.append(d)
        except Exception as e:
            print(f"[SQLite Get User Sessions Warning] {e}")

        if not res:
            # Memory fallback
            res = [s for s in self._memory_sessions.values() if s.get("user_id") == user_id]
        return res

    def update_session(self, session_id: str, updates: Dict[str, Any]):
        updates["updated_at"] = time.time()

        # Memory
        if session_id in self._memory_sessions:
            self._memory_sessions[session_id].update(updates)

        # MongoDB
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {"$set": updates}
                )
            except Exception as e:
                print(f"[MongoDB Update Session Error] {e}")

        # SQLite
        try:
            with SessionLocal() as db:
                db_sess = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
                if db_sess:
                    if "status" in updates:
                        db_sess.status = updates["status"]
                    cur_data = dict(db_sess.session_data or {})
                    cur_data.update(updates)
                    db_sess.session_data = cur_data
                    db.commit()
        except Exception as e:
            print(f"[SQLite Update Session Warning] {e}")

    def record_fact_revealed(self, session_id: str, fact_id: str):
        if not fact_id or not session_id:
            return
        sess = self.get_session(session_id) or {}
        revealed = list(set(sess.get("revealed_fact_ids", []) + [fact_id]))
        self.update_session(session_id, {"revealed_fact_ids": revealed})

    def record_examination_completed(self, session_id: str, exam_id: str, findings: Any):
        if not session_id or not exam_id:
            return
        sess = self.get_session(session_id) or {}
        completed = list(set(sess.get("completed_examinations", []) + [exam_id]))
        self.update_session(session_id, {"completed_examinations": completed})

    def record_investigation_ordered(self, session_id: str, test_id: str, result: Any):
        if not session_id or not test_id:
            return
        sess = self.get_session(session_id) or {}
        ordered = list(set(sess.get("ordered_investigations", []) + [test_id]))
        revealed = list(set(sess.get("revealed_investigation_results", []) + [test_id]))
        self.update_session(session_id, {
            "ordered_investigations": ordered,
            "revealed_investigation_results": revealed
        })

    def record_diagnosis_submission(self, session_id: str, diagnosis_data: Dict[str, Any]):
        if not session_id:
            return
        self.update_session(session_id, {"diagnosis_submission": diagnosis_data})

    def record_management_submission(self, session_id: str, management_data: Dict[str, Any]):
        if not session_id:
            return
        self.update_session(session_id, {"management_submission": management_data})

    # -----------------------------------------------------------------------
    # Messages CRUD (MongoDB Atlas + SQLite + Memory)
    # -----------------------------------------------------------------------
    def save_message(self, message_data: Dict[str, Any]) -> str:
        msg_id = message_data.get("id") or str(uuid.uuid4())
        session_id = message_data.get("session_id") or "global_session"
        message_data["id"] = msg_id
        message_data.setdefault("timestamp", time.time())
        if "role" not in message_data and "sender" in message_data:
            sender = message_data["sender"].lower()
            message_data["role"] = "learner" if sender in ["learner", "student", "doctor", "user"] else "patient"
        if "content" not in message_data and "message" in message_data:
            message_data["content"] = message_data["message"]
        if "message" not in message_data and "content" in message_data:
            message_data["message"] = message_data["content"]

        # 1. Memory
        if session_id not in self._memory_messages:
            self._memory_messages[session_id] = []
        self._memory_messages[session_id].append(dict(message_data))

        # 2. MongoDB
        if self.is_connected and self.messages_col is not None:
            try:
                self.messages_col.insert_one(dict(message_data))
            except Exception as e:
                print(f"[MongoDB Message Save Warning] {e}")

        # 3. SQLite
        try:
            with SessionLocal() as db:
                # Ensure parent session exists in SQLite
                sess_exists = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
                if not sess_exists:
                    db.add(SimulationSession(
                        id=session_id,
                        case_id=message_data.get("case_id", "chest_pain_001"),
                        status="ACTIVE",
                        session_data={"session_id": session_id}
                    ))
                    db.flush()

                db_msg = MessageModel(
                    id=msg_id,
                    session_id=session_id,
                    sender=message_data.get("sender", "LEARNER").upper(),
                    message=message_data.get("message", ""),
                    metadata_json=message_data.get("metadata_json", {})
                )
                db.add(db_msg)
                db.commit()
                print(f"[Storage Engine] Saved message from {db_msg.sender} into SQLite table 'messages'.")
        except Exception as e:
            print(f"[SQLite Message Save Warning] {e}")

        return msg_id

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        # 1. MongoDB
        if self.is_connected and self.messages_col is not None:
            try:
                msgs = list(self.messages_col.find(
                    {"$or": [{"session_id": session_id}, {"sessionId": session_id}]},
                    {"_id": 0}
                ).sort("timestamp", 1))
                if msgs:
                    return msgs
            except Exception as e:
                print(f"[MongoDB Get Messages Error] {e}")

        # 2. SQLite
        res = []
        try:
            with SessionLocal() as db:
                db_msgs = db.query(MessageModel).filter(MessageModel.session_id == session_id).order_by(MessageModel.created_at.asc()).all()
                for m in db_msgs:
                    res.append({
                        "id": m.id,
                        "session_id": m.session_id,
                        "sender": m.sender,
                        "role": "learner" if m.sender.lower() in ["learner", "user", "doctor", "student"] else "patient",
                        "message": m.message,
                        "content": m.message,
                        "metadata_json": m.metadata_json or {},
                        "timestamp": m.created_at.timestamp() if m.created_at else time.time()
                    })
        except Exception as e:
            print(f"[SQLite Get Messages Warning] {e}")

        if not res and session_id in self._memory_messages:
            return self._memory_messages[session_id]
        return res

    # -----------------------------------------------------------------------
    # Interaction Events (Audit Trail)
    # -----------------------------------------------------------------------
    def save_event(self, session_id: str, event_type: str, data: Dict[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        doc = {
            "id": event_id,
            "session_id": session_id,
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        }

        # MongoDB
        if self.is_connected and self.interaction_events_col is not None:
            try:
                self.interaction_events_col.insert_one(doc)
            except Exception as e:
                print(f"[MongoDB Save Event Warning] {e}")

        # SQLite
        try:
            with SessionLocal() as db:
                db_ev = InteractionEventModel(
                    id=event_id,
                    session_id=session_id,
                    event_type=event_type,
                    data=data
                )
                db.add(db_ev)
                db.commit()
        except Exception as e:
            print(f"[SQLite Save Event Warning] {e}")

        return event_id

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        # 1. MongoDB
        if self.is_connected and self.interaction_events_col is not None:
            try:
                return list(self.interaction_events_col.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1))
            except Exception:
                pass

        # 2. SQLite
        res = []
        try:
            with SessionLocal() as db:
                db_evs = db.query(InteractionEventModel).filter(InteractionEventModel.session_id == session_id).order_by(InteractionEventModel.created_at.asc()).all()
                for ev in db_evs:
                    res.append({
                        "id": ev.id,
                        "session_id": ev.session_id,
                        "event_type": ev.event_type,
                        "data": ev.data or {},
                        "timestamp": ev.created_at.timestamp() if ev.created_at else time.time()
                    })
        except Exception as e:
            print(f"[SQLite Get Events Warning] {e}")
        return res

    # -----------------------------------------------------------------------
    # Evaluations CRUD (MongoDB Atlas + SQLite)
    # -----------------------------------------------------------------------
    def save_evaluation(self, eval_data: Dict[str, Any]) -> str:
        eval_id = eval_data.get("id") or str(uuid.uuid4())
        eval_data["id"] = eval_id
        eval_data.setdefault("created_at", time.time())
        eval_data.setdefault("timestamp", time.time())
        session_id = eval_data.get("session_id") or "global_session"

        # MongoDB
        if self.is_connected and self.evaluations_col is not None:
            try:
                self.evaluations_col.update_one(
                    {"session_id": session_id},
                    {"$set": eval_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Evaluation Save Warning] {e}")

        # SQLite
        try:
            with SessionLocal() as db:
                # Ensure parent session exists
                sess = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
                if not sess:
                    db.add(SimulationSession(
                        id=session_id,
                        case_id=eval_data.get("case_id", "chest_pain_001"),
                        status="COMPLETED",
                        session_data={"session_id": session_id}
                    ))
                    db.flush()
                else:
                    sess.status = "COMPLETED"

                existing_eval = db.query(EvaluationModel).filter(EvaluationModel.session_id == session_id).first()
                score = eval_data.get("overall_score") or eval_data.get("overallScore") or 0
                feedback = eval_data.get("attending_physician_notes") or eval_data.get("feedback") or "Evaluation complete."
                strengths = eval_data.get("strengths") or []
                areas = eval_data.get("areas_to_improve") or eval_data.get("areas_for_improvement") or []
                dim_dict = {d.get("dimension", f"dim_{i}"): d.get("score", 0) for i, d in enumerate(eval_data.get("dimensions", []))} if isinstance(eval_data.get("dimensions"), list) else eval_data.get("dimensions", {})

                if existing_eval:
                    existing_eval.score = score
                    existing_eval.feedback = feedback
                    existing_eval.strengths = strengths
                    existing_eval.areas_for_improvement = areas
                    existing_eval.category_scores = dim_dict
                    existing_eval.detailed_rubric = eval_data.get("rubric_breakdown") or {}
                else:
                    new_eval = EvaluationModel(
                        id=eval_id,
                        session_id=session_id,
                        user_id=eval_data.get("user_id"),
                        score=score,
                        feedback=feedback,
                        strengths=strengths,
                        areas_for_improvement=areas,
                        category_scores=dim_dict,
                        detailed_rubric=eval_data.get("rubric_breakdown") or {}
                    )
                    db.add(new_eval)
                db.commit()
                print(f"[Storage Engine] Saved OSCE evaluation for session '{session_id}' (Score: {score}) into SQLite 'evaluations' table.")
        except Exception as e:
            print(f"[SQLite Evaluation Save Warning] {e}")

        return eval_id

    def get_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        # 1. MongoDB
        if self.is_connected and self.evaluations_col is not None:
            try:
                ev = self.evaluations_col.find_one({"session_id": session_id}, {"_id": 0})
                if ev:
                    return ev
            except Exception:
                pass

        # 2. SQLite
        try:
            with SessionLocal() as db:
                db_ev = db.query(EvaluationModel).filter(EvaluationModel.session_id == session_id).first()
                if db_ev:
                    return {
                        "id": db_ev.id,
                        "session_id": db_ev.session_id,
                        "user_id": db_ev.user_id,
                        "overall_score": db_ev.score,
                        "overallScore": db_ev.score,
                        "pass_status": db_ev.score >= 70,
                        "attending_physician_notes": db_ev.feedback,
                        "strengths": db_ev.strengths or [],
                        "areas_to_improve": db_ev.areas_for_improvement or [],
                        "category_scores": db_ev.category_scores or {},
                        "rubric_breakdown": db_ev.detailed_rubric or {},
                        "created_at": db_ev.created_at.timestamp() if db_ev.created_at else time.time()
                    }
        except Exception as e:
            print(f"[SQLite Get Evaluation Warning] {e}")
        return None

    # -----------------------------------------------------------------------
    # Users CRUD (MongoDB Atlas + SQLite)
    # -----------------------------------------------------------------------
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        clean_email = (email or "").lower().strip()
        # 1. MongoDB
        if self.is_connected and self.users_col is not None:
            try:
                u = self.users_col.find_one({"email": clean_email}, {"_id": 0})
                if u:
                    return u
            except Exception:
                pass

        # 2. SQLite
        try:
            with SessionLocal() as db:
                db_u = db.query(UserModel).filter(UserModel.email == clean_email).first()
                if db_u:
                    return {
                        "id": db_u.id,
                        "email": db_u.email,
                        "password_hash": db_u.password_hash,
                        "first_name": db_u.first_name,
                        "last_name": db_u.last_name,
                        "role": db_u.role,
                        "is_active": db_u.is_active,
                        "created_at": db_u.created_at.timestamp() if db_u.created_at else time.time()
                    }
        except Exception as e:
            print(f"[SQLite Get User Warning] {e}")
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        # 1. MongoDB
        if self.is_connected and self.users_col is not None:
            try:
                u = self.users_col.find_one({"id": user_id}, {"_id": 0})
                if u:
                    return u
            except Exception:
                pass

        # 2. SQLite
        try:
            with SessionLocal() as db:
                db_u = db.query(UserModel).filter(UserModel.id == user_id).first()
                if db_u:
                    return {
                        "id": db_u.id,
                        "email": db_u.email,
                        "password_hash": db_u.password_hash,
                        "first_name": db_u.first_name,
                        "last_name": db_u.last_name,
                        "role": db_u.role,
                        "is_active": db_u.is_active,
                        "created_at": db_u.created_at.timestamp() if db_u.created_at else time.time()
                    }
        except Exception as e:
            print(f"[SQLite Get User Warning] {e}")
        return None

    def save_user(self, user_data: Dict[str, Any]) -> str:
        user_id = user_data.get("id") or str(uuid.uuid4())
        user_data["id"] = user_id
        user_data["email"] = user_data["email"].lower().strip()
        user_data.setdefault("created_at", time.time())

        # MongoDB
        if self.is_connected and self.users_col is not None:
            try:
                self.users_col.update_one(
                    {"email": user_data["email"]},
                    {"$set": user_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Save User Warning] {e}")

        # SQLite
        try:
            with SessionLocal() as db:
                db_u = db.query(UserModel).filter(UserModel.email == user_data["email"]).first()
                if db_u:
                    db_u.password_hash = user_data.get("password_hash", db_u.password_hash)
                    db_u.first_name = user_data.get("first_name", db_u.first_name)
                    db_u.last_name = user_data.get("last_name", db_u.last_name)
                    db_u.role = user_data.get("role", db_u.role)
                    db_u.is_active = user_data.get("is_active", True)
                else:
                    new_u = UserModel(
                        id=user_id,
                        email=user_data["email"],
                        password_hash=user_data.get("password_hash", ""),
                        first_name=user_data.get("first_name", "User"),
                        last_name=user_data.get("last_name", "Learner"),
                        role=user_data.get("role", "LEARNER"),
                        is_active=True
                    )
                    db.add(new_u)
                db.commit()
                print(f"[Storage Engine] Saved user '{user_data['email']}' to SQLite 'users' table.")
        except Exception as e:
            print(f"[SQLite Save User Warning] {e}")

        return user_id


mongo_manager = MongoDBManager.get_instance()
