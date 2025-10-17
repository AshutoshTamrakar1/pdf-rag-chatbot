import pyttsx3
from faster_whisper import WhisperModel
import speech_recognition as sr
from logging_config import get_logger, log_exceptions

logger = get_logger(__name__)

def generate_tts_audio(text: str, output_path: str):
    """Generate speech audio from text using pyttsx3 (local TTS)."""
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

def transcribe_audio_local(audio_path: str) -> str:
    """Transcribe audio to text using speech_recognition (local STT)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_sphinx(audio)  # or use recognize_google(audio)

def generate_podcast_audio_local(script: str, output_path: str):
    """Generate podcast audio from script using local TTS."""
    generate_tts_audio(script, output_path)