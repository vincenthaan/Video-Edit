"""faster-whisper로 단어 단위 타임스탬프를 얻는다."""

from __future__ import annotations

from .models import Word

# Whisper는 "어", "음" 같은 간투사를 지워버리는 경향이 있다.
# 간투사가 섞인 문장을 프롬프트로 주면 들리는 그대로 받아적을 확률이 높아진다.
VERBATIM_PROMPTS = {
    "ko": "음, 어... 그러니까 제가, 어, 그 그 말씀드리고 싶은 건요, 음, 이거예요.",
    "en": "Um, so, uh, I I wanted to, like, uh, talk about this.",
    "ja": "えーと、あの、その、えー、つまりですね。",
}


def transcribe(
    path: str,
    model_size: str = "large-v3-turbo",
    language: str | None = "ko",
    device: str = "auto",
    compute_type: str = "default",
    verbatim: bool = True,
    progress: bool = True,
) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("faster-whisper가 필요합니다: pip install faster-whisper") from e

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    prompt = VERBATIM_PROMPTS.get(language or "", None) if verbatim else None
    segments, info = model.transcribe(
        path,
        language=language,
        word_timestamps=True,
        initial_prompt=prompt,
        # 이전 문맥에 끌려가 같은 문장을 반복 생성(환각)하는 현상을 줄인다.
        condition_on_previous_text=False,
        # VAD 필터를 켜면 짧은 간투사까지 날아가므로 끈다. 무음은 따로 잡는다.
        vad_filter=False,
    )

    words: list[Word] = []
    total = info.duration or 0
    for seg in segments:
        for w in seg.words or []:
            if not w.word.strip():
                continue
            words.append(Word(w.word, float(w.start), float(w.end), float(w.probability)))
        if progress and total:
            print(f"\r  음성 인식 중... {min(seg.end / total, 1):5.1%}", end="", flush=True)
    if progress:
        print()
    return words
