from sqlalchemy import Column, Integer, String
from models.base import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, unique=True, nullable=True)
    user_name = Column(String, nullable=False)
    birth_year = Column(Integer)
    guardian_name = Column(String)

    reports = relationship("ParentReport", back_populates="user")