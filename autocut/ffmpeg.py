"""ffmpeg 실행 헬퍼. 시스템 ffmpeg가 없으면 imageio-ffmpeg 번들 바이너리를 사용한다."""

from __future__ import annotations

import re
import shutil
import subprocess
from fractions import Fraction
from functools import lru_cache


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하거나 `pip install imageio-ffmpeg` 하세요."
        ) from e


@lru_cache(maxsize=1)
def ffmpeg_major_version() -> int:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
    m = re.search(r"ffmpeg version n?(\d+)", out)
    return int(m.group(1)) if m else 0


def run(args: list[str]) -> subprocess.CompletedProcess:
    """ffmpeg를 실행하고 실패 시 stderr 마지막 부분과 함께 예외를 던진다."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg 실패 (exit {proc.returncode}):\n{tail}")
    return proc


def media_duration(path: str) -> float:
    """ffprobe 없이 ffmpeg 출력에서 길이를 읽는다."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise RuntimeError(f"영상 길이를 읽을 수 없습니다: {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def video_fps(path: str) -> Fraction | None:
    """첫 번째 영상 스트림의 프레임레이트. 영상이 없으면 None."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stderr.splitlines():
        if re.search(r"Stream #\S+.*: Video:", line) and "attached pic" not in line:
            m = re.search(r"([\d.]+) fps", line) or re.search(r"([\d.]+) tbr", line)
            return _to_fraction(float(m.group(1))) if m else Fraction(30)
    return None


def _to_fraction(fps: float) -> Fraction:
    for ntsc in (24, 30, 60, 120):
        if abs(fps - ntsc * 1000 / 1001) < 0.01:
            return Fraction(ntsc * 1000, 1001)
    return Fraction(fps).limit_denominator(1001)
