from sqlalchemy import Column, Integer, ForeignKey, DateTime
from datetime import datetime
from app.models.base import Base

class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    session_id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    step_id = Column(Integer, ForeignKey("steps.step_id"), nullable=False)

    words_count = Column(Integer, nullable=False)

    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
