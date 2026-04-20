from sqlalchemy import Column, Integer, ForeignKey, Text, DateTime, String
from datetime import datetime
from .base import Base
from sqlalchemy.orm import relationship

class ParentReport(Base):
    __tablename__ = "parent_reports"

    report_id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    period_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    ment = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")