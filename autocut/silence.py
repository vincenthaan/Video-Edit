"""ffmpeg silencedetect로 무음 구간을 찾는다."""

from __future__ import annotations

import re

from . import ffmpeg

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silencedetect(stderr: str, duration: float) -> list[tuple[float, float]]:
    """silencedetect 로그를 (start, end) 목록으로 변환한다."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        m = _START.search(line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = _END.search(line)
        if m and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    if start is not None:  # 파일 끝까지 무음
        spans.append((start, duration))
    return spans


def detect_silence(
    path: str, duration: float, noise_db: float = -35.0, min_silence: float = 0.5
) -> list[tuple[float, float]]:
    proc = ffmpeg.run(
        [
            "-i", path,
            "-vn",
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ]
    )
    return parse_silencedetect(proc.stderr, duration)
