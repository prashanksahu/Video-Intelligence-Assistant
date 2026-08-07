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

def transcribe_single_chunk(chunk_path : str, translate : bool = False) -> str:
    model = load_whisper_model() 

    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path, task = task)

    return result['text']

def transcribe_all(chunk_paths : list, translate : bool = False) -> str:
    full_transcript = ""

    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}...")
        text = transcribe_single_chunk(chunk_path = chunk_path, translate = translate)

        full_transcript += text +" "

    print("Transcription completed...")

    return full_transcript