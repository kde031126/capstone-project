from sqlalchemy import Column, Integer, String
from app.models.base import Base

class Step(Base):
    __tablename__ = "steps"

    step_id = Column(Integer, primary_key=True, index=True)
    step_name = Column(String(100), nullable=False)
    step_order = Column(Integer, nullable=False)
