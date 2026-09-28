"""남은 단어로 자막 줄을 만들고 SRT로 저장한다."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Cue, Word
from .timeline import TimeMap


@dataclass
class SubtitleOptions:
    max_chars: int = 18  # 한 줄 최대 글자 수 (공백 제외). 쇼츠면 12~15 추천
    max_duration: float = 4.0
    break_gap: float = 0.5  # 단어 사이가 이만큼 벌어지면 줄을 나눈다 (원본 기준)
    min_duration: float = 0.5
    keep_punct: bool = False  # False면 마침표·쉼표 제거 (? ! 는 유지)


_SENTENCE_END = re.compile(r"[.?!。？！]$")
_DROP_PUNCT = re.compile(r"[.,。、…]+")


def _clean(text: str, keep_punct: bool) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not keep_punct:
        text = _DROP_PUNCT.sub("", text).strip()
    return text


def _chars(text: str) -> int:
    return len(text.replace(" ", ""))


def build_cues(
    words: list[Word], removed: set[int], tmap: TimeMap, opts: SubtitleOptions | None = None
) -> list[Cue]:
    opts = opts or SubtitleOptions()
    # 편집 후에도 절반 이상 살아있는 단어만 자막에 쓴다
    kept = [
        w for i, w in enumerate(words)
        if i not in removed and tmap.kept_ratio(w.start, w.end) >= 0.5
    ]

    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in kept:
        if cur:
            text = "".join(x.text for x in cur) + w.text
            too_long = _chars(text) > opts.max_chars
            too_slow = w.end - cur[0].start > opts.max_duration
            paused = w.start - cur[-1].end > opts.break_gap
            ended = bool(_SENTENCE_END.search(cur[-1].text.strip()))
            if too_long or too_slow or paused or ended:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    cues: list[Cue] = []
    for g in groups:
        text = _clean("".join(w.text for w in g), opts.keep_punct)
        if not text:
            continue
        start, end = tmap.map(g[0].start), tmap.map(g[-1].end)
        cues.append(Cue(start, max(end, start + 0.01), text))

    # 너무 짧은 자막은 다음 자막 시작 전까지 늘리고, 겹침은 없앤다
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else tmap.total
        if c.end - c.start < opts.min_duration:
            c.end = min(c.start + opts.min_duration, nxt)
        c.end = min(c.end, nxt)
    return [c for c in cues if c.end > c.start]


def _ts(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)
