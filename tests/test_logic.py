import pytest

from autocut.disfluency import find_disfluencies
from autocut.models import Cut, Word
from autocut.silence import parse_silencedetect
from autocut.subtitles import build_cues, to_srt
from autocut.timeline import (
    CutOptions,
    TimeMap,
    disfluency_cuts,
    keep_segments,
    merge_cuts,
    silence_cuts,
    snap_keeps,
)


def seq(*items, gap=0.05, dur=0.3):
    """("단어", ...) 또는 ("단어", 쉼) 목록으로 연속된 Word 목록을 만든다."""
    words, t = [], 0.0
    for it in items:
        if isinstance(it, (int, float)):
            t += it
            continue
        words.append(Word(" " + it, t, t + dur))
        t += dur + gap
    return words


def spans(cuts):
    return [t for c in cuts for t in (c.start, c.end)]


def flat(pairs):
    return [t for p in pairs for t in p]


def texts(words, marks):
    return [(words[i].text.strip(), r) for i, r in sorted(marks.items())]


def test_fillers():
    w = seq("어", "오늘은", "음...", "날씨가", "어어", "좋네요")
    assert texts(w, find_disfluencies(w)) == [("어", "filler"), ("음...", "filler"), ("어어", "filler")]


def test_soft_filler_needs_pause():
    # 쉼 없이 이어진 "그 사람"의 "그"는 지시어라 남긴다
    assert find_disfluencies(seq("그", "사람이", "왔어요")) == {}
    # "그... 사람" 처럼 쉬면 버벅임
    w = seq("그", 0.4, "사람이", "왔어요")
    assert texts(w, find_disfluencies(w)) == [("그", "filler")]


def test_repeats_keep_last():
    w = seq("그래서", "그래서", "그래서", "제가")
    assert texts(w, find_disfluencies(w)) == [("그래서", "repeat"), ("그래서", "repeat")]


def test_repeat_phrase_across_filler():
    w = seq("제가", "오늘", "어", "제가", "오늘", "말씀드릴")
    marks = find_disfluencies(w)
    assert marks == {0: "repeat", 1: "repeat", 2: "filler"}


def test_repeat_far_apart_is_intentional():
    w = seq("좋아요", 1.5, "좋아요")
    assert find_disfluencies(w) == {}


def test_fragment():
    w = seq("안녕", "안녕하세요", "그러", "그러니까")
    assert texts(w, find_disfluencies(w)) == [("안녕", "fragment"), ("그러", "fragment")]


def test_disfluency_cut_removes_surrounding_pause():
    w = [Word(" 오늘", 0.0, 0.5), Word(" 어", 1.0, 1.2), Word(" 날씨", 1.8, 2.2)]
    cuts = disfluency_cuts(w, {1: "filler"}, pad=0.1)
    assert spans(cuts) == pytest.approx(flat([(0.6, 1.7)]))


def test_parse_silencedetect():
    log = (
        "[silencedetect @ 0x1] silence_start: -0.01\n"
        "[silencedetect @ 0x1] silence_end: 1.2 | silence_duration: 1.2\n"
        "[silencedetect @ 0x1] silence_start: 5.5\n"
    )
    assert parse_silencedetect(log, 8.0) == [(0.0, 1.2), (5.5, 8.0)]


def test_silence_cut_padding_and_merge():
    cuts = silence_cuts([(0.0, 1.2), (3.0, 4.0)], pad=0.1)
    assert spans(cuts) == pytest.approx(flat([(0.0, 1.1), (3.1, 3.9)]))
    merged = merge_cuts(cuts + [Cut(3.5, 4.5, "filler")], 10.0, CutOptions())
    assert spans(merged) == pytest.approx(flat([(0.0, 1.1), (3.1, 4.5)]))


def test_merge_swallows_tiny_keep():
    merged = merge_cuts([Cut(1.0, 2.0, "silence"), Cut(2.05, 3.0, "filler")], 10, CutOptions())
    assert [(c.start, c.end) for c in merged] == [(1.0, 3.0)]


def test_timemap():
    keeps = keep_segments([Cut(1.0, 2.0, "silence"), Cut(3.0, 4.0, "silence")], 5.0)
    assert keeps == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    tm = TimeMap(keeps)
    assert tm.total == 3.0
    assert tm.map(0.5) == 0.5
    assert tm.map(1.5) == 1.0  # 잘린 구간 → 다음 구간 시작
    assert tm.map(4.5) == 2.5
    assert tm.kept_ratio(1.0, 2.0) == 0.0


def test_snap_keeps():
    assert snap_keeps([(0.0, 1.01), (2.49, 3.0)], 10) == [(0.0, 1.0), (2.5, 3.0)]


def test_cues_and_srt():
    w = seq("안녕하세요.", "어", "오늘은", "날씨가", "좋네요")
    marks = find_disfluencies(w)
    tm = TimeMap([(0, 100)])
    cues = build_cues(w, set(marks), tm)
    assert [c.text for c in cues] == ["안녕하세요", "오늘은 날씨가 좋네요"]
    srt = to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,")
    assert "2\n" in srt


def test_cues_split_by_length():
    w = seq("가나다라마바", "사아자차카타", "파하가나다라", "마바사")
    cues = build_cues(w, set(), TimeMap([(0, 100)]))
    assert all(len(c.text.replace(" ", "")) <= 18 for c in cues)
    assert len(cues) == 2
