"""공용 데이터 구조."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Word:
    """음성 인식된 단어 하나 (원본 영상 기준 시간, 초)."""

    text: str
    start: float
    end: float
    prob: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(d["text"], float(d["start"]), float(d["end"]), float(d.get("prob", 1.0)))


@dataclass
class Cut:
    """잘라낼 구간 (원본 영상 기준 시간, 초)."""

    start: float
    end: float
    reason: str  # silence | filler | repeat | fragment | gap

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        return cls(float(d["start"]), float(d["end"]), d.get("reason", "manual"))


@dataclass
class Cue:
    """자막 한 줄 (컷 편집된 영상 기준 시간, 초)."""

    start: float
    end: float
    text: str
