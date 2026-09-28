"""autocut: 무음·버벅임 자동 컷 + 자막 생성 → CapCut 으로 가져가기."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import ffmpeg
from .disfluency import DisfluencyOptions, find_disfluencies
from .models import Cut, Word
from .silence import detect_silence
from .subtitles import SubtitleOptions, build_cues, to_srt
from .timeline import (
    CutOptions,
    TimeMap,
    disfluency_cuts,
    keep_segments,
    merge_cuts,
    silence_cuts,
    snap_keeps,
    word_gap_cuts,
)

REASON_KO = {
    "silence": "무음",
    "gap": "말 없는 구간",
    "filler": "간투사(어/음..)",
    "repeat": "반복",
    "fragment": "말 끊김",
    "manual": "수동",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="autocut",
        description="무음·버벅이는 구간을 자동으로 잘라내고 자막(SRT)을 만듭니다. "
        "결과 영상과 SRT를 CapCut에 가져오면 됩니다.",
    )
    p.add_argument("input", help="원본 영상/음성 파일")
    p.add_argument("-o", "--outdir", help="결과 폴더 (기본: 원본과 같은 폴더)")

    g = p.add_argument_group("음성 인식")
    g.add_argument("--model", default="large-v3-turbo",
                   help="Whisper 모델 (tiny/base/small/medium/large-v3/large-v3-turbo)")
    g.add_argument("--language", default="ko", help="언어 코드, 자동 감지는 auto")
    g.add_argument("--device", default="auto", help="auto / cpu / cuda")
    g.add_argument("--compute-type", default="default", help="예: int8 (CPU 절약), float16 (GPU)")
    g.add_argument("--retranscribe", action="store_true", help="저장된 인식 결과를 무시하고 다시 인식")

    g = p.add_argument_group("무음 컷")
    g.add_argument("--silence-db", type=float, default=-35.0,
                   help="이 음량(dB)보다 작으면 무음 (기본 -35, 잡음 많으면 -30)")
    g.add_argument("--min-silence", type=float, default=0.5, help="이 길이(초) 이상 무음만 자름")
    g.add_argument("--pad", type=float, default=0.1, help="말 앞뒤로 남길 여유(초)")
    g.add_argument("--max-word-gap", type=float, default=1.0,
                   help="단어 사이 공백이 이보다 길면 자름 (0=끔)")
    g.add_argument("--no-silence", action="store_true", help="무음 컷 끄기")

    g = p.add_argument_group("버벅임 컷")
    g.add_argument("--no-fillers", action="store_true", help="'어/음' 같은 간투사 컷 끄기")
    g.add_argument("--no-soft-fillers", action="store_true",
                   help="쉼 앞뒤의 '그/저/뭐/이제' 컷 끄기")
    g.add_argument("--no-repeats", action="store_true", help="'그래서 그래서' 반복 컷 끄기")
    g.add_argument("--no-fragments", action="store_true", help="'안- 안녕하세요' 말 끊김 컷 끄기")
    g.add_argument("--filler", action="append", default=[], metavar="단어",
                   help="간투사로 취급할 단어 추가 (여러 번 사용 가능)")

    g = p.add_argument_group("자막")
    g.add_argument("--max-chars", type=int, default=18, help="자막 한 줄 최대 글자 수 (쇼츠는 12~15)")
    g.add_argument("--keep-punct", action="store_true", help="자막의 마침표/쉼표 유지")

    g = p.add_argument_group("출력")
    g.add_argument("--cuts", metavar="JSON",
                   help="직접 수정한 cuts.json 으로 다시 렌더링 (음성 인식/분석 생략)")
    g.add_argument("--dry-run", action="store_true", help="분석만 하고 cuts.json만 저장")
    g.add_argument("--no-render", action="store_true", help="영상 렌더링 없이 SRT/cuts.json만 저장")
    g.add_argument("--crf", type=int, default=18, help="화질 (낮을수록 고화질, 기본 18)")
    g.add_argument("--preset", default="medium", help="x264 프리셋 (ultrafast~veryslow)")
    return p.parse_args(argv)


def _fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}"


def analyze(args: argparse.Namespace, src: Path, outdir: Path, duration: float):
    words_path = outdir / f"{src.stem}.words.json"
    if words_path.exists() and not args.retranscribe:
        print(f"• 저장된 음성 인식 결과 사용: {words_path.name}")
        words = [Word.from_dict(d) for d in json.loads(words_path.read_text("utf-8"))]
    else:
        print(f"• 음성 인식 (모델: {args.model}) — 처음엔 모델 다운로드로 시간이 걸립니다")
        from .transcribe import transcribe

        words = transcribe(
            str(src),
            model_size=args.model,
            language=None if args.language == "auto" else args.language,
            device=args.device,
            compute_type=args.compute_type,
        )
        words_path.write_text(
            json.dumps([w.to_dict() for w in words], ensure_ascii=False, indent=1), "utf-8"
        )

    opts = CutOptions(pad=args.pad, max_word_gap=args.max_word_gap or None)
    cuts: list[Cut] = []
    if not args.no_silence:
        print("• 무음 구간 찾는 중")
        spans = detect_silence(str(src), duration, args.silence_db, args.min_silence)
        cuts += silence_cuts(spans, opts.pad)
        if opts.max_word_gap and words:
            cuts += word_gap_cuts(words, opts.max_word_gap, opts.pad)

    dopts = DisfluencyOptions(
        fillers=not args.no_fillers,
        soft_fillers=not args.no_soft_fillers,
        repeats=not args.no_repeats,
        fragments=not args.no_fragments,
        extra_fillers={f.strip().lower() for f in args.filler},
    )
    marks = find_disfluencies(words, dopts)
    cuts += disfluency_cuts(words, marks, opts.pad)
    return words, marks, merge_cuts(cuts, duration, opts)


def save_cuts_json(path: Path, src: Path, duration: float, words, marks, cuts) -> None:
    data = {
        "_help": "cuts 에서 항목을 지우면 그 구간이 살아나고, 추가하면 잘립니다. "
                 "수정 후: autocut 원본 --cuts 이파일.json",
        "source": str(src),
        "duration": round(duration, 3),
        "cuts": [c.to_dict() for c in cuts],
        "words": [{**w.to_dict(), **({"mark": marks[i]} if i in marks else {})}
                  for i, w in enumerate(words)],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        print(f"파일이 없습니다: {src}", file=sys.stderr)
        return 1
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)

    duration = ffmpeg.media_duration(str(src))
    fps = ffmpeg.video_fps(str(src))
    cuts_path = outdir / f"{src.stem}.cuts.json"

    if args.cuts:
        data = json.loads(Path(args.cuts).read_text("utf-8"))
        words = [Word.from_dict(d) for d in data["words"]]
        marks = {i: d["mark"] for i, d in enumerate(data["words"]) if d.get("mark")}
        cuts = merge_cuts([Cut.from_dict(c) for c in data["cuts"]], duration, CutOptions(min_cut=0))
        print(f"• 수정된 컷 목록 사용: {args.cuts} ({len(cuts)}개)")
    else:
        words, marks, cuts = analyze(args, src, outdir, duration)
        save_cuts_json(cuts_path, src, duration, words, marks, cuts)

    keeps = keep_segments(cuts, duration)
    if fps is not None:
        keeps = snap_keeps(keeps, float(fps))
    tmap = TimeMap(keeps)

    counts = Counter(c.reason for c in cuts)
    removed = Counter()
    for c in cuts:
        removed[c.reason] += c.duration
    print(f"\n원본 {_fmt(duration)} → 편집본 {_fmt(tmap.total)} "
          f"({duration - tmap.total:.1f}초 / {len(cuts)}곳 잘림)")
    for reason, n in counts.most_common():
        print(f"  - {REASON_KO.get(reason, reason)}: {n}곳, {removed[reason]:.1f}초")
    if not args.cuts:
        print(f"\n컷 목록: {cuts_path}  (검토·수정 후 --cuts 로 다시 렌더링 가능)")

    if args.dry_run:
        return 0
    if not keeps:
        print("남는 구간이 없습니다. --silence-db 값을 낮춰(예: -45) 보세요.", file=sys.stderr)
        return 1

    # 간투사는 영상에 남더라도 자막에서는 뺀다. 반복/말 끊김은 컷에서 되살렸다면 자막에도 보인다.
    fillers = {i for i, r in marks.items() if r == "filler"}
    cues = build_cues(words, fillers, tmap, SubtitleOptions(
        max_chars=args.max_chars, keep_punct=args.keep_punct))
    ext = ".mp4" if fps is not None else ".m4a"
    out_media = outdir / f"{src.stem}_autocut{ext}"
    out_srt = outdir / f"{src.stem}_autocut.srt"
    out_srt.write_text(to_srt(cues), "utf-8")
    print(f"자막: {out_srt} ({len(cues)}줄)")

    if not args.no_render:
        print("• 렌더링 중...")
        from .render import render

        render(str(src), str(out_media), keeps, fps, crf=args.crf, preset=args.preset)
        print(f"영상: {out_media}")

    print("\nCapCut: 영상을 가져온 뒤 [텍스트 → 자막 가져오기/로컬 자막]에서 SRT를 불러오세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
