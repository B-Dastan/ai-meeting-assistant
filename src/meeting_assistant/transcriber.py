"""Audio transcription using OpenAI Whisper."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import whisper
from dotenv import load_dotenv

load_dotenv()


class Transcriber:
    """Handles audio transcription using Whisper model."""

    def __init__(self, model_size: Optional[str] = None):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size or os.environ["WHISPER_MODEL_SIZE"]
        self._model: Optional[whisper.Whisper] = None

    @property
    def model(self) -> whisper.Whisper:
        """Lazy-load the Whisper model."""
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe_file(self, audio_path: str | Path) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary with 'text' (full transcript) and 'segments' (timestamped chunks).
        """
        audio_path = str(audio_path)
        result: dict = self.model.transcribe(
            audio_path, fp16=False)  # type: ignore[assignment]
        segments = result.get("segments", [])
        return {
            "text": str(result.get("text", "")).strip(),
            "segments": [
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": str(seg.get("text", "")).strip(),
                }
                for seg in segments
            ],
            "language": result.get("language", "en"),
        }

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Transcribe audio from a numpy array.

        Args:
            audio_array: Audio data as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary with 'text' and 'segments'.
        """
        # Save to a temporary file (Whisper requires file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            return self.transcribe_file(tmp.name)

    @staticmethod
    def save_audio(audio_array: np.ndarray, path: str | Path, sample_rate: int = 16000) -> Path:
        """
        Save audio array to a WAV file.

        Args:
            audio_array: Audio data as numpy array.
            path: Output file path.
            sample_rate: Sample rate of the audio.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio_array, sample_rate)
        return path
