"""남길 구간만 이어붙여 새 영상을 만든다.

구간마다 trim 을 거는 방식은 컷이 수백 개면 영상을 수백 번 디코딩하므로,
select/aselect 로 한 번에 걸러내는 단일 패스 방식을 쓴다.
"""

from __future__ import annotations

import os
import tempfile
from fractions import Fraction

from . import ffmpeg


def _select_expr(keeps: list[tuple[float, float]], shift: float = 0.0) -> str:
    # 반열린 구간 [s, e) 을 써야 경계 프레임이 두 번 들어가지 않는다.
    return "+".join(f"gte(t,{s - shift:.6f})*lt(t,{e - shift:.6f})" for s, e in keeps)


def build_filtergraph(keeps: list[tuple[float, float]], fps: Fraction | None) -> str:
    # 오디오는 64샘플(약 1.3ms) 단위로 쪼개 선택 경계를 정밀하게 맞춘다.
    audio = (
        "[0:a:0]asetpts=PTS-STARTPTS,asetnsamples=n=64:p=0,"
        f"aselect='{_select_expr(keeps)}',asetpts=N/SR/TB[a]"
    )
    if fps is None:
        return audio
    # fps 필터로 가변 프레임(VFR, 폰 촬영본에 흔함)을 고정 프레임으로 만든 뒤,
    # 프레임 중앙을 기준으로 선택해 부동소수 오차로 프레임이 빠지는 것을 막는다.
    half = float(1 / fps) / 2
    video = (
        f"[0:v:0]setpts=PTS-STARTPTS,fps={fps.numerator}/{fps.denominator},"
        f"select='{_select_expr(keeps, shift=half)}',setpts=N/FRAME_RATE/TB[v]"
    )
    return f"{video};{audio}"


def render(
    src: str,
    dst: str,
    keeps: list[tuple[float, float]],
    fps: Fraction | None,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    graph = build_filtergraph(keeps, fps)
    fd, script = tempfile.mkstemp(suffix=".txt", prefix="autocut_filter_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(graph)
    try:
        # ffmpeg 7부터 -filter_complex_script 는 폐기 예정 → -/filter_complex 사용
        script_opt = ["-/filter_complex", script] if ffmpeg.ffmpeg_major_version() >= 7 else [
            "-filter_complex_script", script
        ]
        args = ["-y", "-i", src, *script_opt]
        if fps is not None:
            args += [
                "-map", "[v]", "-r", f"{fps.numerator}/{fps.denominator}",
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            ]
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst]
        ffmpeg.run(args)
    finally:
        os.unlink(script)
