"""
Case and Clinical Entity ORM models.
Includes normalized tables for Cases, Patient Profiles, Clinical Facts,
Physical Findings, Investigations, and Hidden Evaluation data.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, Boolean, DateTime, Integer, JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"CS-{uuid.uuid4().hex[:6].upper()}")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False, default="Beginner")
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Normalized 1-to-1 and 1-to-Many Relationships
    patient_profile = relationship("PatientProfile", back_populates="case", uselist=False, cascade="all, delete-orphan")
    clinical_facts = relationship("ClinicalFact", back_populates="case", uselist=False, cascade="all, delete-orphan")
    physical_findings = relationship("PhysicalFinding", back_populates="case", cascade="all, delete-orphan")
    investigations = relationship("Investigation", back_populates="case", cascade="all, delete-orphan")
    hidden_evaluation = relationship("CaseHiddenEvaluation", back_populates="case", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("SimulationSession", back_populates="case", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Case id={self.id} title='{self.title}'>"


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"pt-{uuid.uuid4().hex[:8]}")
    case_id: Mapped[str] = mapped_column(String(50), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    occupation: Mapped[str] = mapped_column(String(150), nullable=True)
    persona: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    case = relationship("Case", back_populates="patient_profile")


class ClinicalFact(Base):
    __tablename__ = "clinical_facts"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"cf-{uuid.uuid4().hex[:8]}")
    case_id: Mapped[str] = mapped_column(String(50), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=True)
    history: Mapped[str] = mapped_column(Text, nullable=True)
    onset: Mapped[str] = mapped_column(Text, nullable=True)
    timing: Mapped[str] = mapped_column(Text, nullable=True)
    location: Mapped[str] = mapped_column(Text, nullable=True)
    character: Mapped[str] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(Text, nullable=True)
    radiation: Mapped[str] = mapped_column(Text, nullable=True)
    aggravating_factors: Mapped[str] = mapped_column(Text, nullable=True)
    relieving_factors: Mapped[str] = mapped_column(Text, nullable=True)
    associated_symptoms: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    past_medical_history: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    medications: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    allergies: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    family_history: Mapped[str] = mapped_column(Text, nullable=True)
    social_history: Mapped[str] = mapped_column(Text, nullable=True)
    smoking: Mapped[str] = mapped_column(Text, nullable=True)
    alcohol: Mapped[str] = mapped_column(Text, nullable=True)
    facts_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=True)

    case = relationship("Case", back_populates="clinical_facts")


class PhysicalFinding(Base):
    __tablename__ = "physical_findings"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"pf-{uuid.uuid4().hex[:8]}")
    case_id: Mapped[str] = mapped_column(String(50), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    system: Mapped[str] = mapped_column(String(50), nullable=False) # cardiovascular, respiratory, abdominal, neurological, general, HEENT
    finding: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)

    case = relationship("Case", back_populates="physical_findings")


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"inv-{uuid.uuid4().hex[:8]}")
    case_id: Mapped[str] = mapped_column(String(50), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    result: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=True)
    reference_range: Mapped[str] = mapped_column(String(100), nullable=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    case = relationship("Case", back_populates="investigations")


class CaseHiddenEvaluation(Base):
    """
    CRITICAL: Never returned through standard learner case APIs.
    Only available to the evaluation engine and administrative users.
    """
    __tablename__ = "case_hidden_evaluations"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"che-{uuid.uuid4().hex[:8]}")
    case_id: Mapped[str] = mapped_column(String(50), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    diagnosis: Mapped[str] = mapped_column(String(255), nullable=False)
    differential_diagnosis: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    management: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    learning_objectives: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    scoring_rubric: Mapped[dict] = mapped_column(JSON, default=dict, nullable=True)

    case = relationship("Case", back_populates="hidden_evaluation")
