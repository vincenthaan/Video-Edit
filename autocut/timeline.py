"""컷 구간 계산과 원본↔편집본 시간 변환."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Cut, Word


@dataclass
class CutOptions:
    pad: float = 0.1  # 말 앞뒤로 남겨둘 여유 (초)
    min_cut: float = 0.12  # 이보다 짧은 컷은 무시 (너무 잘게 자르면 부자연스러움)
    min_keep: float = 0.08  # 이보다 짧게 남는 조각은 같이 잘라낸다
    max_word_gap: float | None = 1.0  # 단어 사이 공백이 이보다 길면 잘라낸다 (None=끔)


def silence_cuts(spans: list[tuple[float, float]], pad: float) -> list[Cut]:
    cuts = []
    for s, e in spans:
        # 영상 맨 앞/뒤 무음은 여유 없이 통째로 자른다
        cs = s if s <= 0.01 else s + pad
        cuts.append(Cut(cs, e - pad, "silence"))
    return [c for c in cuts if c.duration > 0]


def word_gap_cuts(words: list[Word], max_gap: float, pad: float) -> list[Cut]:
    """음성 인식상 아무 말도 없는 긴 구간 (배경 소음 때문에 무음 감지가 놓친 곳)."""
    cuts = []
    for a, b in zip(words, words[1:]):
        if b.start - a.end > max_gap:
            cuts.append(Cut(a.end + pad, b.start - pad, "gap"))
    return cuts


def disfluency_cuts(words: list[Word], marks: dict[int, str], pad: float) -> list[Cut]:
    """표시된 단어(연속된 묶음 단위)와 그 주변 쉼을 잘라낸다."""
    cuts = []
    i = 0
    while i < len(words):
        if i not in marks:
            i += 1
            continue
        j = i
        while j + 1 < len(words) and (j + 1) in marks:
            j += 1
        reasons = [marks[k] for k in range(i, j + 1)]
        reason = max(set(reasons), key=reasons.count)
        # 앞뒤 남는 단어 사이의 쉼까지 함께 잘라 "어..." 뒤의 멈칫함도 없앤다.
        start = words[i - 1].end + pad if i > 0 else words[i].start
        end = words[j + 1].start - pad if j + 1 < len(words) else words[j].end
        start = min(start, words[i].start)
        end = max(end, words[j].end)
        cuts.append(Cut(start, end, reason))
        i = j + 1
    return cuts


def merge_cuts(cuts: list[Cut], duration: float, opts: CutOptions) -> list[Cut]:
    """겹치는 컷을 합치고, 너무 짧은 컷/조각을 정리한다."""
    items = sorted(
        (Cut(max(0.0, c.start), min(duration, c.end), c.reason) for c in cuts),
        key=lambda c: c.start,
    )
    merged: list[Cut] = []
    for c in items:
        if c.duration <= 0:
            continue
        if merged and c.start - merged[-1].end < opts.min_keep:
            last = merged[-1]
            if c.end > last.end:
                reason = last.reason if last.duration >= c.duration else c.reason
                merged[-1] = Cut(last.start, c.end, reason)
        else:
            merged.append(c)
    # 끝에 짧은 조각이 남으면 같이 자르기
    if merged and duration - merged[-1].end < opts.min_keep:
        merged[-1] = Cut(merged[-1].start, duration, merged[-1].reason)
    if merged and merged[0].start < opts.min_keep:
        merged[0] = Cut(0.0, merged[0].end, merged[0].reason)
    return [c for c in merged if c.duration >= opts.min_cut]


def keep_segments(cuts: list[Cut], duration: float) -> list[tuple[float, float]]:
    keeps = []
    t = 0.0
    for c in cuts:
        if c.start > t:
            keeps.append((t, c.start))
        t = max(t, c.end)
    if t < duration:
        keeps.append((t, duration))
    return keeps


class TimeMap:
    """원본 시간을 편집본 시간으로 변환한다."""

    def __init__(self, keeps: list[tuple[float, float]]):
        self.keeps = keeps
        self.offsets = []
        acc = 0.0
        for s, e in keeps:
            self.offsets.append(acc)
            acc += e - s
        self.total = acc

    def map(self, t: float) -> float:
        """t가 잘린 구간 안이면 다음 남는 구간의 시작으로 붙인다."""
        for (s, e), off in zip(self.keeps, self.offsets):
            if t < s:
                return off
            if t <= e:
                return off + (t - s)
        return self.total

    def kept_ratio(self, start: float, end: float) -> float:
        """[start, end] 중 남아 있는 비율."""
        if end <= start:
            return 1.0 if any(s <= start <= e for s, e in self.keeps) else 0.0
        kept = sum(max(0.0, min(e, end) - max(s, start)) for s, e in self.keeps)
        return kept / (end - start)


def snap_keeps(keeps: list[tuple[float, float]], fps: float) -> list[tuple[float, float]]:
    """경계를 프레임 격자에 맞춘다. 영상/음성 길이가 컷마다 조금씩 어긋나 싱크가 밀리는 것을 막는다."""
    snapped = []
    for s, e in keeps:
        s2, e2 = round(s * fps) / fps, round(e * fps) / fps
        if e2 > s2:
            snapped.append((s2, e2))
    return snapped

