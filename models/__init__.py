"""Models package — imports all ORM models so Alembic and SQLAlchemy detect them."""

from .user import User
from .case_model import (
    Case,
    PatientProfile,
    ClinicalFact,
    PhysicalFinding,
    Investigation,
    CaseHiddenEvaluation
)
from .session_model import SimulationSession
from .message import Message
from .evaluation import Evaluation

__all__ = [
    "User",
    "Case",
    "PatientProfile",
    "ClinicalFact",
    "PhysicalFinding",
    "Investigation",
    "CaseHiddenEvaluation",
    "SimulationSession",
    "Message",
    "Evaluation",
]
