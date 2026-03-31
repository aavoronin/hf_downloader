from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelErrorLog:
    model_name: str
    error_count: int = 0
    last_error_date: Optional[str] = None
    last_error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelErrorLog':
        return cls(**data)
