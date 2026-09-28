"""버벅임(간투사·반복·말 끊김) 단어를 찾는다.

규칙 기반이라 설명 가능하고, 결과는 cuts.json 으로 남아 사람이 검토·수정할 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Word

# 항상 간투사로 보는 단어 (정규화 후 비교)
FILLERS = {
    "어", "음", "으", "으음", "에", "아", "흠", "엄", "오",
    "uh", "um", "umm", "uhm", "er", "erm", "ah", "hmm", "mm",
    "えー", "えーと", "あの", "えっと",
}
# "어어어", "음음", "으으음" 처럼 간투사 음절만 반복된 단어
_FILLER_RUN = re.compile(r"^(?:어|음|으|에|아|흠|엄)+$")

# 의미가 있을 때도 있는 말 버릇: 앞이나 뒤에 쉼(pause)이 있을 때만 간투사로 본다.
# 예) "그 사람"(지시어) vs "그... 사람"(버벅임)
SOFT_FILLERS = {"그", "저", "뭐", "이제", "막", "약간", "뭐지", "like"}

_STRIP = re.compile(r"[\s\.,!?…~\-·'\"“”‘’()\[\]]+")


def normalize(text: str) -> str:
    return _STRIP.sub("", text).lower()


@dataclass
class DisfluencyOptions:
    fillers: bool = True
    soft_fillers: bool = True
    repeats: bool = True
    fragments: bool = True
    soft_filler_pause: float = 0.25  # 이만큼 쉬면 SOFT_FILLERS 도 간투사로 판단
    repeat_max_gap: float = 1.0  # 반복 사이 간격이 이보다 길면 의도된 반복으로 본다
    max_ngram: int = 3
    extra_fillers: set[str] = field(default_factory=set)


def _is_filler(norm: str, extra: set[str]) -> bool:
    return norm in FILLERS or norm in extra or bool(_FILLER_RUN.match(norm))


def find_disfluencies(words: list[Word], opts: DisfluencyOptions | None = None) -> dict[int, str]:
    """제거할 단어 인덱스 → 이유(filler/repeat/fragment)."""
    opts = opts or DisfluencyOptions()
    norms = [normalize(w.text) for w in words]
    marks: dict[int, str] = {}

    for i, n in enumerate(norms):
        if not n:
            continue
        if opts.fillers and _is_filler(n, opts.extra_fillers):
            marks[i] = "filler"
        elif opts.soft_fillers and n in SOFT_FILLERS:
            gap_before = words[i].start - words[i - 1].end if i > 0 else 0.0
            gap_after = words[i + 1].start - words[i].end if i + 1 < len(words) else 0.0
            if max(gap_before, gap_after) >= opts.soft_filler_pause:
                marks[i] = "filler"

    # 반복: "그래서 그래서", "제가 오늘 제가 오늘" → 마지막 것만 남긴다.
    # (말을 다시 시작할 때 보통 마지막 시도가 가장 온전하다)
    if opts.repeats:
        # 간투사를 건너뛴 인덱스 목록 위에서 비교해야 "그래서 어 그래서"도 잡힌다.
        live = [i for i in range(len(words)) if norms[i] and i not in marks]
        changed = True
        while changed:
            changed = False
            for n in range(opts.max_ngram, 0, -1):
                k = 0
                while k + 2 * n <= len(live):
                    a = live[k : k + n]
                    b = live[k + n : k + 2 * n]
                    gap = words[b[0]].start - words[a[-1]].end
                    if [norms[i] for i in a] == [norms[i] for i in b] and gap <= opts.repeat_max_gap:
                        for i in a:
                            marks[i] = "repeat"
                        live = live[:k] + live[k + n :]
                        changed = True
                    else:
                        k += 1

    # 말 끊김: "안- 안녕하세요", "그러 그러니까" → 앞 조각 제거
    if opts.fragments:
        live = [i for i in range(len(words)) if norms[i] and i not in marks]
        for a, b in zip(live, live[1:]):
            na, nb = norms[a], norms[b]
            gap = words[b].start - words[a].end
            if len(na) < len(nb) and nb.startswith(na) and gap <= opts.repeat_max_gap:
                # 한 글자 조각은 오탐이 많으니 "-" 로 끊겼거나 쉼이 있을 때만
                if len(na) >= 2 or words[a].text.rstrip().endswith(("-", "…", "...")) or gap >= 0.15:
                    marks[a] = "fragment"

    return marks
