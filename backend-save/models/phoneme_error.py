from sqlalchemy import Column, Integer, ForeignKey, String
from models.base import Base
from sqlalchemy.orm import relationship

class PhonemeError(Base):
    __tablename__ = "phoneme_errors"

    error_id = Column(Integer, primary_key=True, index=True)

    record_id = Column(Integer, ForeignKey("records.record_id"), nullable=False)

    target_phoneme = Column(String(50), nullable=True)
    error_phoneme = Column(String(50), nullable=True)

    error_type = Column(String(50), nullable=True)
    error_position = Column(Integer, nullable=True)

    record = relationship("Record", back_populates="errors")