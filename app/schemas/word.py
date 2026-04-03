from pydantic import BaseModel, ConfigDict
from typing import Optional

class WordBase(BaseModel):
    word_text: str
    standard_phonemes: str
    image_url: str
    audio_url: Optional[str] = None

class WordResponse(WordBase):
    word_id: int

    model_config = ConfigDict(from_attributes=True)