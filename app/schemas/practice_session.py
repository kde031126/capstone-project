from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class SessionCreate(BaseModel):
    user_id: int
    step_id: int
    words_count: int

class SessionResponse(SessionCreate):
    session_id: int
    started_at: datetime
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)