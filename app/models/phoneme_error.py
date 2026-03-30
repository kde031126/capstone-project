from sqlalchemy import Column, Integer, ForeignKey, String
from app.models.base import Base

class PhonemeError(Base):
    __tablename__ = "phoneme_errors"

    error_id = Column(Integer, primary_key=True, index=True)

    record_id = Column(Integer, ForeignKey("records.record_id"), nullable=False)

    target_phoneme = Column(String(50), nullable=False)
    error_phoneme = Column(String(50), nullable=False)

    error_type = Column(String(50), nullable=False)
    error_position = Column(Integer, nullable=False)
