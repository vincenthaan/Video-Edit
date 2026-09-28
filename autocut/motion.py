"""화면 변화(필기, 마우스, 손 움직임 등)가 있는 구간을 찾는다.

말은 없어도 화면에서 뭔가 하고 있는 구간은 무음이라도 자르면 안 되기 때문에,
작게 줄인 흑백 프레임끼리 비교해서 "바뀐 픽셀 비율"이 일정 이상인 구간을 표시한다.
"""

from __future__ import annotations

import subprocess

from . import ffmpeg

W, H = 320, 180  # 필기 획처럼 가는 변화도 남도록 너무 작게 줄이지 않는다


def frame_activity(path: str, sample_fps: float = 5.0, pixel_thresh: int = 20) -> list[int]:
    """샘플링한 프레임마다 직전 프레임 대비 바뀐 픽셀 수 (320x180 기준). i번째 값은 [i/fps, (i+1)/fps] 구간.

    축소 과정에서 카메라 노이즈는 평균되어 사라지고, 가만히 있는 화면은 거의 0이 나온다.
    """
    import numpy as np

    proc = subprocess.Popen(
        [
            ffmpeg.ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", path, "-an",
            "-vf", f"setpts=PTS-STARTPTS,fps={sample_fps},scale={W}:{H},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE,
    )
    size = W * H
    counts: list[int] = []
    prev = None
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
        if prev is not None:
            counts.append(int(np.count_nonzero(np.abs(cur - prev) > pixel_thresh)))
        prev = cur
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("화면 변화 분석 중 ffmpeg 오류")
    return counts


def active_spans(
    counts: list[int],
    sample_fps: float = 5.0,
    min_pixels: int = 6,
    join_gap: float = 1.0,
    pad: float = 0.3,
) -> list[tuple[float, float]]:
    """바뀐 픽셀이 min_pixels 이상인 구간. 가까운 구간은 합치고 앞뒤로 여유를 둔다.

    필기 한 획은 320x180 으로 줄이면 몇~수십 픽셀 정도라 기본값을 낮게 잡는다.
    """
    step = 1.0 / sample_fps
    spans: list[tuple[float, float]] = []
    for i, n in enumerate(counts):
        if n < min_pixels:
            continue
        s, e = i * step, (i + 1) * step
        if spans and s - spans[-1][1] <= join_gap:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    return [(max(0.0, s - pad), e + pad) for s, e in spans]


def detect_activity(path: str, min_pixels: int = 6) -> list[tuple[float, float]]:
    return active_spans(frame_activity(path), min_pixels=min_pixels)
