from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List
from .phoneme_error import PhonemeErrorResponse

class RecordCreate(BaseModel):
    session_id: int
    word_id: int
    child_text: Optional[str] = None
    child_phonemes: str

class RecordResponse(RecordCreate):
    record_id: int
    is_correct: bool 
    created_at: datetime
    errors : List[PhonemeErrorResponse] = []
    model_config = ConfigDict(from_attributes=True)