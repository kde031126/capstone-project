from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List
from .phoneme_error import PhonemeErrorResponse

class RecordCreate(BaseModel):
    session_id: int
    word_id: int
    child_text: Optional[str] = None
    child_phonemes: Optional[str] = None # Optional 추가

class RecordResponse(RecordCreate):
    record_id: int
    is_correct: bool 
    created_at: datetime
    # 여기에 session_id와 word_id가 상속되어 있으므로, 
    # 아래 2번 단계의 return 문에서 이 값들을 꼭 넣어줘야 합니다.
    errors : List[PhonemeErrorResponse] = []
    
    model_config = ConfigDict(from_attributes=True)