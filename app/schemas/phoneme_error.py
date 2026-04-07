from pydantic import BaseModel, ConfigDict
from typing import Optional

class PhonemeErrorBase(BaseModel):
    record_id: int
    target_phoneme: Optional[str] = None
    error_phoneme: Optional[str] = None
    error_type: Optional[str] = None
    error_position: Optional[int] = None

class PhonemeErrorResponse(PhonemeErrorBase):
    error_id: int

    model_config = ConfigDict(from_attributes=True)