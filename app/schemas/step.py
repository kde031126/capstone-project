from pydantic import BaseModel, ConfigDict
from typing import List
from .word import WordResponse

class StepBase(BaseModel):
    step_name: str
    step_order: int

class StepResponse(StepBase):
    step_id: int
    words: List[WordResponse] = []

    model_config = ConfigDict(from_attributes=True)