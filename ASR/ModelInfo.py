from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class ModelInfo:
    name: str
    folder_name: str
    size_bytes: int
    size_human: str
    path: Path
    files: List[str]

    def to_dict(self) -> dict:
        return asdict(self)
