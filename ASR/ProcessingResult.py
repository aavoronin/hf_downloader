from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ProcessingResult:
    model_name: str
    text: Optional[str]
    success: bool
    time_taken: float
    datetime_completed: str
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
