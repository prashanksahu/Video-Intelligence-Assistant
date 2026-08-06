import whisper
import os
from rich import print

WHISPER_MODEL = os.getenv("WHISPER_MODEL", default='small')

_model = None

def load_whisper_model():
    global _model

    if _model is None:
        print(f"Loading model...")
        _model = whisper.load_model(name = WHISPER_MODEL)
        print("Whisper model loaded successfully...")

    return _model
