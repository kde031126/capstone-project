from sqlalchemy import Column, Integer, ForeignKey, Text, DateTime
from datetime import datetime
from app.models.base import Base

class Record(Base):
    __tablename__ = "records"

    record_id = Column(Integer, primary_key=True, index=True)

    session_id = Column(Integer, ForeignKey("practice_sessions.session_id"), nullable=False)
    word_id = Column(Integer, ForeignKey("words.word_id"), nullable=False)

    child_text = Column(Text, nullable=True)
    child_phonemes = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
