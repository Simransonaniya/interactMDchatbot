"""
InteractMD — Production MongoDB Database Connector & Repository.
Authoritative source of truth for clinical cases, simulation sessions, messages,
interaction events, and evaluations.
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from config import settings


class MongoDBManager:
    _instance: Optional["MongoDBManager"] = None

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.is_connected = False
        self._init_connection()

    @classmethod
    def get_instance(cls) -> "MongoDBManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_connection(self):
        uri = settings.MONGODB_URI or settings.DATABASE_URL
        if not uri or not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
            print("[MongoDB Warning] No valid MongoDB URI configured. Running in offline/fallback mode.")
            return

        try:
            db_name = settings.MONGODB_DB_NAME or "interactmd"
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client.get_database(db_name)
            self.is_connected = True
            print("========================================================")
            print(f"  [MongoDB Atlas] Connected to live cluster: {db_name}")
            print("========================================================")
            self._ensure_indexes()
        except Exception as e:
            print(f"[MongoDB Connection Warning] Could not connect to Atlas ({e}). Operating in resilient mode.")
            self.is_connected = False

    def _ensure_indexes(self):
        if not self.is_connected or self.db is None:
            return
        try:
            # cases
            self.cases_col.create_index("case_id", unique=True)
            self.cases_col.create_index("id")
            self.cases_col.create_index("specialty")
            self.cases_col.create_index("status")

            # sessions
            self.sessions_col.create_index("session_id", unique=True)
            self.sessions_col.create_index("id")
            self.sessions_col.create_index("user_id")
            self.sessions_col.create_index("case_id")

            # messages
            self.messages_col.create_index("session_id")
            self.messages_col.create_index([("session_id", 1), ("timestamp", 1)])

            # interaction events
            self.interaction_events_col.create_index("session_id")
            self.interaction_events_col.create_index([("session_id", 1), ("timestamp", 1)])

            # evaluations
            self.evaluations_col.create_index("session_id", unique=True)
            self.evaluations_col.create_index("user_id")

            # users
            self.users_col.create_index("email", unique=True)
            self.users_col.create_index("id", unique=True)
        except Exception as e:
            print(f"[MongoDB Index Warning] {e}")

    # -----------------------------------------------------------------------
    # Collections
    # -----------------------------------------------------------------------
    @property
    def cases_col(self) -> Collection:
        if self.db is not None:
            return self.db["cases"]
        return None

    @property
    def sessions_col(self) -> Collection:
        if self.db is not None:
            return self.db["sessions"]
        return None

    @property
    def messages_col(self) -> Collection:
        if self.db is not None:
            return self.db["messages"]
        return None

    @property
    def interaction_events_col(self) -> Collection:
        if self.db is not None:
            return self.db["interaction_events"]
        return None

    @property
    def evaluations_col(self) -> Collection:
        if self.db is not None:
            return self.db["evaluations"]
        return None

    @property
    def users_col(self) -> Collection:
        if self.db is not None:
            return self.db["users"]
        return None

    # -----------------------------------------------------------------------
    # Cases CRUD
    # -----------------------------------------------------------------------
    def get_all_cases(self, specialty: Optional[str] = None, status: str = "published") -> List[Dict[str, Any]]:
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
        return []

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.cases_col is not None:
            try:
                # Support matching by case_id or legacy id
                c = self.cases_col.find_one({"$or": [{"case_id": case_id}, {"id": case_id}]}, {"_id": 0})
                if c:
                    return c
            except Exception as e:
                print(f"[MongoDB Query Error] {e}")
        return None

    def save_case(self, case_data: Dict[str, Any]) -> str:
        case_id = case_data.get("case_id") or case_data.get("id") or str(uuid.uuid4())
        case_data["case_id"] = case_id
        if "id" not in case_data:
            case_data["id"] = case_id
        if "version" not in case_data:
            case_data["version"] = 1
        if "updated_at" not in case_data:
            case_data["updated_at"] = time.time()

        if self.is_connected and self.cases_col is not None:
            try:
                self.cases_col.update_one(
                    {"$or": [{"case_id": case_id}, {"id": case_id}]},
                    {"$set": case_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Case Save Error] {e}")
        return case_id

    # -----------------------------------------------------------------------
    # Sessions CRUD
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

        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.insert_one(dict(session_data))
            except Exception as e:
                print(f"[MongoDB Session Create Error] {e}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.sessions_col is not None:
            try:
                return self.sessions_col.find_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {"_id": 0}
                )
            except Exception as e:
                print(f"[MongoDB Get Session Error] {e}")
        return None

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        if self.is_connected and self.sessions_col is not None:
            try:
                return list(self.sessions_col.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1))
            except Exception as e:
                print(f"[MongoDB Get User Sessions Error] {e}")
        return []

    def update_session(self, session_id: str, updates: Dict[str, Any]):
        updates["updated_at"] = time.time()
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {"$set": updates}
                )
            except Exception as e:
                print(f"[MongoDB Update Session Error] {e}")

    def record_fact_revealed(self, session_id: str, fact_id: str):
        if not fact_id:
            return
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {
                        "$addToSet": {"revealed_fact_ids": fact_id},
                        "$set": {"updated_at": time.time()}
                    }
                )
            except Exception as e:
                print(f"[MongoDB Record Fact Error] {e}")

    def record_examination_completed(self, session_id: str, exam_id: str, findings: Any):
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {
                        "$addToSet": {"completed_examinations": exam_id},
                        "$set": {"updated_at": time.time()}
                    }
                )
            except Exception as e:
                print(f"[MongoDB Record Exam Error] {e}")

    def record_investigation_ordered(self, session_id: str, test_id: str, result: Any):
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {
                        "$addToSet": {
                            "ordered_investigations": test_id,
                            "revealed_investigation_results": test_id
                        },
                        "$set": {"updated_at": time.time()}
                    }
                )
            except Exception as e:
                print(f"[MongoDB Record Investigation Error] {e}")

    def record_diagnosis_submission(self, session_id: str, diagnosis_data: Dict[str, Any]):
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {
                        "$set": {
                            "diagnosis_submission": diagnosis_data,
                            "updated_at": time.time()
                        }
                    }
                )
            except Exception as e:
                print(f"[MongoDB Save Diagnosis Error] {e}")

    def record_management_submission(self, session_id: str, management_data: Dict[str, Any]):
        if self.is_connected and self.sessions_col is not None:
            try:
                self.sessions_col.update_one(
                    {"$or": [{"session_id": session_id}, {"id": session_id}]},
                    {
                        "$set": {
                            "management_submission": management_data,
                            "updated_at": time.time()
                        }
                    }
                )
            except Exception as e:
                print(f"[MongoDB Save Management Error] {e}")

    # -----------------------------------------------------------------------
    # Messages CRUD
    # -----------------------------------------------------------------------
    def save_message(self, message_data: Dict[str, Any]) -> str:
        msg_id = message_data.get("id") or str(uuid.uuid4())
        message_data["id"] = msg_id
        message_data.setdefault("timestamp", time.time())
        if "role" not in message_data and "sender" in message_data:
            sender = message_data["sender"].lower()
            message_data["role"] = "learner" if sender in ["learner", "student", "doctor", "user"] else "patient"
        if "content" not in message_data and "message" in message_data:
            message_data["content"] = message_data["message"]

        if self.is_connected and self.messages_col is not None:
            try:
                self.messages_col.insert_one(dict(message_data))
            except Exception as e:
                print(f"[MongoDB Message Save Warning] {e}")
        return msg_id

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if self.is_connected and self.messages_col is not None:
            try:
                return list(self.messages_col.find(
                    {"$or": [{"session_id": session_id}, {"sessionId": session_id}]},
                    {"_id": 0}
                ).sort("timestamp", 1))
            except Exception as e:
                print(f"[MongoDB Get Messages Error] {e}")
        return []

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
        if self.is_connected and self.interaction_events_col is not None:
            try:
                self.interaction_events_col.insert_one(doc)
            except Exception as e:
                print(f"[MongoDB Save Event Warning] {e}")
        return event_id

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        if self.is_connected and self.interaction_events_col is not None:
            try:
                return list(self.interaction_events_col.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1))
            except Exception:
                pass
        return []

    # -----------------------------------------------------------------------
    # Evaluations CRUD
    # -----------------------------------------------------------------------
    def save_evaluation(self, eval_data: Dict[str, Any]) -> str:
        eval_id = eval_data.get("id") or str(uuid.uuid4())
        eval_data["id"] = eval_id
        eval_data.setdefault("created_at", time.time())
        eval_data.setdefault("timestamp", time.time())

        session_id = eval_data.get("session_id")
        if self.is_connected and self.evaluations_col is not None:
            try:
                self.evaluations_col.update_one(
                    {"session_id": session_id},
                    {"$set": eval_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Evaluation Save Warning] {e}")
        return eval_id

    def get_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.evaluations_col is not None:
            try:
                return self.evaluations_col.find_one({"session_id": session_id}, {"_id": 0})
            except Exception:
                pass
        return None

    # -----------------------------------------------------------------------
    # Users CRUD
    # -----------------------------------------------------------------------
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.users_col is not None:
            try:
                return self.users_col.find_one({"email": email.lower().strip()}, {"_id": 0})
            except Exception:
                pass
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.users_col is not None:
            try:
                return self.users_col.find_one({"id": user_id}, {"_id": 0})
            except Exception:
                pass
        return None

    def save_user(self, user_data: Dict[str, Any]) -> str:
        user_id = user_data.get("id") or str(uuid.uuid4())
        user_data["id"] = user_id
        user_data["email"] = user_data["email"].lower().strip()
        user_data.setdefault("created_at", time.time())

        if self.is_connected and self.users_col is not None:
            try:
                self.users_col.update_one(
                    {"email": user_data["email"]},
                    {"$set": user_data},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Save User Warning] {e}")
        return user_id


mongo_manager = MongoDBManager.get_instance()

