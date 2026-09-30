"""Punctuated, sentence-level transcription for the director (faster-whisper).

Why these settings (measured 2026-09-29 on the 188 s voice-clone clip, analysis/whisper_punct_test.py):
  sentence endings per 30 s window
  large-v3, cond_on_prev=False (Graydient TranscribeAudioFromURL)  0,0,0,0,4,0,0   <- the unpunctuated run
  large-v3 + style prompt                                          3,1,2,2,0,0,0
  medium, cond_on_prev=True                                        2,0,2,0,0,0,1   <- conditioning copies the
                                                                                      unpunctuated style forward
  medium + style prompt, cond_on_prev=False  (chosen)              4,4,5,7,8,4,2   (34 vs 36 real sentences)
medium is also ~2x faster than large-v3. VAD did not help (28). A style prompt with "!" turned statements
into exclamations, so it only models periods, commas and a question mark.

Output keeps TranscribeAudioFromURL's lyrics_json shape (timeline of lyric entries with words), one entry
per SENTENCE, so RigStudioIntentPrompt / RigStudioIntentPack consume it unchanged.
"""
import re

STYLE_PROMPT = "Hello, and welcome back. Today, I'll walk you through it, step by step. Sound good? Let's begin."
SETTINGS = dict(beam_size=5, word_timestamps=True, no_speech_threshold=0.7, condition_on_previous_text=False)
_END = re.compile(r"[.?!]['\"]?$")


def load_model(model_dir_or_size):
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(model_dir_or_size, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(model_dir_or_size, device="cpu", compute_type="int8")


def read_wav16k(path):
    """16 kHz mono s16 WAV (from runtime.fetch_audio_16k) -> float32 samples. Handing faster-whisper an
    array skips its PyAV decoder: on Graydient's 2026-09-30 image, PyAV rejects the
    av.open(metadata_errors=...) call faster-whisper makes (render on rig-direct v2)."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(model, wav_path, language=None, style_prompt=STYLE_PROMPT):
    audio = read_wav16k(wav_path) if isinstance(wav_path, str) else wav_path
    segs, info = model.transcribe(audio, language=language or None, initial_prompt=style_prompt, **SETTINGS)
    words = [w for s in segs for w in (s.words or []) if w.word.strip()]
    timeline, cur = [], []

    def flush():
        if cur:
            timeline.append({"start": round(cur[0].start, 2), "end": round(cur[-1].end, 2),
                             "text": " ".join(w.word.strip() for w in cur), "type": "lyric",
                             "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "text": w.word.strip()}
                                       for w in cur]})
            cur.clear()

    for w in words:
        # a long silence also ends a sentence, even without punctuation
        if cur and w.start - cur[-1].end > 1.2:
            flush()
        cur.append(w)
        if _END.search(w.word.strip()):
            flush()
    flush()
    return {"language": info.language, "duration": round(info.duration, 2), "timeline": timeline,
            "transcriber": {"engine": "faster-whisper", "style_prompt": style_prompt, **SETTINGS}}
