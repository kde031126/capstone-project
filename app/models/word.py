from sqlalchemy import Column, Integer, String, Text
from app.models.base import Base

class Word(Base):
    __tablename__ = "words"

    word_id = Column(Integer, primary_key=True, index=True)
    word_text = Column(String(100), nullable=False)
    standard_phonemes = Column(Text, nullable=False)
