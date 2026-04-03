from pydantic import BaseModel, ConfigDict

class PhonemeErrorBase(BaseModel):
    record_id: int
    target_phoneme: str
    error_phoneme: str
    error_type: str
    error_position: int

class PhonemeErrorResponse(PhonemeErrorBase):
    error_id: int

    model_config = ConfigDict(from_attributes=True)