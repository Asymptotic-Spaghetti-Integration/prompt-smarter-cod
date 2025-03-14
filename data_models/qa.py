from pydantic import BaseModel
from typing import List, Optional, Union
from enum import Enum


class QAModel(BaseModel):
    question: str
    options: List[str]
    answer: str
    source: str
    tags: Optional[List[str]] = None
    explanation: Optional[str] = None