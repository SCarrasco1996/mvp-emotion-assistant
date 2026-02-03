"""Configuración central del MVP.

Se centralizan rutas, parámetros y nombres de modelos utilizados por el pipeline:
STT (Whisper), SER (clasificador de emociones), LLM (Ollama) y TTS (pyttsx3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# Directorio raíz del proyecto (mvp-emotion-assistant).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    """Parámetros globales del MVP."""

    # Rutas base del proyecto.
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    # Dispositivo preferente para cómputo acelerado.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- STT ----------
    # Modelos de Whisper disponibles: tiny, base, small, medium, large-v2, large-v3.
    whisper_model_name: str = "large-v3"
    # Precisión por defecto (se ajusta en __post_init__ según el dispositivo).
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 5

    # ---------- SER ----------
    # Encoder preentrenado base para el ajuste fino de emociones.
    ser_encoder_name: str = "microsoft/wavlm-large"
    ser_sampling_rate: int = 16000

    # Mapeo de las 5 emociones definidas en el sistema.
    ser_label2id: dict = field(
        default_factory=lambda: {
            "alegria": 0,
            "ira": 1,
            "tristeza": 2,
            "miedo": 3,
            "neutro": 4,
        }
    )
    ser_id2label: dict = field(init=False)

    # Ruta donde se guarda/carga el modelo SER.
    ser_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "model")

    # ---------- Respuestas / LLM ----------
    use_ollama: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_temperature: float = 0.6
    ollama_top_p: float = 0.8
    ollama_max_tokens: int = 160
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "30m"

    # ---------- TTS ----------
    enable_tts: bool = True
    tts_voice: str | None = None

    # ---------- Conversación ----------
    # Número de pares usuario/asistente conservados en el historial.
    max_history_turns: int = 10
    # Umbral para considerar baja confianza en la emoción detectada.
    emotion_low_confidence_threshold: float = 0.35

    # ---------- Audio de entrada ----------
    # Duración máxima permitida (segundos) para la interfaz y las validaciones.
    audio_max_duration_s: float = 20.0
    # Duración mínima para evitar fragmentos vacíos o casi vacíos.
    audio_min_duration_s: float = 0.9

    # Control de la interfaz: conservar o limpiar el audio tras enviarlo.
    keep_audio_after_send: bool = False

    # ---------- Varios ----------
    debug: bool = True

    def __post_init__(self) -> None:
        # Se deriva el mapeo inverso para SER.
        self.ser_id2label = {v: k for k, v in self.ser_label2id.items()}

        # Se garantiza la existencia del directorio de datos.
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # En CPU se fuerza float32 para evitar incompatibilidades con float16.
        if self.device != "cuda" and self.whisper_compute_type == "float16":
            self.whisper_compute_type = "float32"

        # Se normalizan los límites de audio para evitar configuraciones inconsistentes.
        if self.audio_min_duration_s < 0:
            self.audio_min_duration_s = 0.0
        if self.audio_max_duration_s < self.audio_min_duration_s:
            self.audio_max_duration_s = self.audio_min_duration_s

        if self.debug:
            print(
                f"[Settings] dispositivo={self.device}, whisper_compute_type={self.whisper_compute_type}"
            )

settings = Settings()
