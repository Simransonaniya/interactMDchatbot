"""
Message ORM model.
Stores each dialogue turn within a simulation session.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("simulation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    sender: Mapped[str] = mapped_column(String(20), nullable=False) # LEARNER, PATIENT, SYSTEM
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)

    # Relationship
    session = relationship("SimulationSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} session_id={self.session_id} sender={self.sender}>"
