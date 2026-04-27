from sqlalchemy import Column, Integer, ForeignKey
from models.base import Base

class StepWord(Base):
    __tablename__ = "step_words"

    step_words_id = Column(Integer, primary_key=True, index=True)

    step_id = Column(Integer, ForeignKey("steps.step_id"), nullable=False)
    word_id = Column(Integer, ForeignKey("words.word_id"), nullable=False)

    word_order = Column(Integer, nullable=False)
