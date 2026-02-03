"""Reconocimiento de voz (STT) con Whisper.

Se utiliza `faster-whisper` como backend para la transcripción en español,
midiendo latencia por invocación.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from .config import settings


class SpeechToTextWhisper:

    def __init__(self) -> None:
        self.model_name = settings.whisper_model_name
        self.device = "cuda" if settings.device == "cuda" else "cpu"
        self.compute_type = settings.whisper_compute_type
        self.beam_size = max(1, int(settings.whisper_beam_size))

        # En GPU se restringe a tipos de cómputo soportados.
        if self.device == "cuda" and self.compute_type not in {"float16", "float32"}:
            self.compute_type = "float16"

        if settings.debug:
            print(
                f"[STT] Cargando WhisperModel '{self.model_name}' en {self.device} "
                f"(compute_type={self.compute_type}, beam_size={self.beam_size})"
            )

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """Transcribe un archivo de audio a texto en español.

        Devuelve un diccionario con el texto reconocido, metadatos básicos y la
        latencia de la transcripción.
        """
        audio_path = Path(audio_path)
        t0 = time.perf_counter()

        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            language="es",  # Se fuerza el reconocimiento en español por estabilidad.
        )

        text = "".join(segment.text for segment in segments).strip()
        latency = time.perf_counter() - t0

        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "latency": latency,
        }


if __name__ == "__main__":
    # Ejecución manual de prueba desde consola de comandos.
    stt = SpeechToTextWhisper()
    print(stt.transcribe("ruta/a/un/audio.wav"))
