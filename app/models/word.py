from sqlalchemy import Column, Integer, String, Text, ForeignKey
from app.models.base import Base

class Word(Base):
    __tablename__ = "words"

    word_id = Column(Integer, primary_key=True, index=True)
    word_text = Column(String(100), nullable=False)

    step_id = Column(Integer, ForeignKey("steps.step_id"), nullable=False)
    standard_phonemes = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    audio_url = Column(String, nullable=False)