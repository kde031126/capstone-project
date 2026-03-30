from pydantic import BaseModel

class UserCreate(BaseModel):
    user_name: str
    birth_year: int
    guardian_name: str

class UserResponse(BaseModel):
    user_id: int
    user_name: str
    birth_year: int
    guardian_name: str

    class Config:
        orm_mode = True
