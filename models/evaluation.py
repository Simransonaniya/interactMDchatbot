"""
Evaluation ORM model.
Stores feedback and performance breakdown for completed simulation sessions.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, Float, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("simulation_sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feedback: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list] = mapped_column(JSON, default=list)
    areas_for_improvement: Mapped[list] = mapped_column(JSON, default=list)
    category_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    detailed_rubric: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("SimulationSession", back_populates="evaluation")
    user = relationship("User", back_populates="evaluations")

    def __repr__(self) -> str:
        return f"<Evaluation id={self.id} session_id={self.session_id} score={self.score}>"
