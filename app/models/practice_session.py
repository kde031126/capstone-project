from sqlalchemy import Column, Integer, ForeignKey, DateTime, JSON
from datetime import datetime
from app.models.base import Base
from sqlalchemy.orm import relationship

class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    session_id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    step_id = Column(Integer, ForeignKey("steps.step_id"), nullable=False)

    target_words = Column(JSON, nullable=False)
    words_count = Column(Integer, nullable=False)

    is_completed = Column(Integer, default=0) # 0: 진행중, 1: 완료

    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    records = relationship("Record", back_populates="session")