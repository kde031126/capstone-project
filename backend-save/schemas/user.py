from pydantic import BaseModel, ConfigDict
from typing import Optional

class UserCreate(BaseModel):
    user_name: str
    birth_year: int
    guardian_name: str
    firebase_uid: Optional[str] = None

class UserResponse(BaseModel):
    user_id: int
    user_name: str
    birth_year: int
    guardian_name: str
    firebase_uid: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)