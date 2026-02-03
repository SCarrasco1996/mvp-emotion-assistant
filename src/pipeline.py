"""Orquestación del pipeline E2E.

Implementa el flujo principal del MVP:
    Audio -> STT (Whisper) -> SER (clasificador de emoción) -> Generación de respuesta (LLM/plantilla) -> TTS (pyttsx3)

Se incluyen dos modos:
- `process`: ejecución puntual (compatibilidad y evaluación).
- `process_turn`: modo conversacional con historial acotado.
"""

from __future__ import annotations

import tempfile
import time
import uuid
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pyttsx3

from .config import settings
from .emotion_ser import EmotionPrediction, EmotionSER
from .responses import (
    ChatResponseResult,
    ResponseResult,
    chat_responder,
    generate_response,
)
from .stt_whisper import SpeechToTextWhisper


TTS_TIMEOUT_SECONDS = 8.0
TTS_LOCK_TIMEOUT_SECONDS = 1.0


@dataclass
class PipelineResult:
    transcript: str
    emotion: str
    emotion_score: float
    response_text: str
    response_audio_path: Optional[str]
    latencies: Dict[str, float]
    scores_per_class: Dict[str, float]
    used_ollama: bool


@dataclass
class ChatPipelineResult:
    transcript: str
    emotion: str
    emotion_score: float
    assistant_text: str
    response_audio_path: Optional[str]
    latencies: Dict[str, float]
    scores_per_class: Dict[str, float]
    used_ollama: bool
    history: List[Dict[str, str]]


class EmotionAssistantPipeline:
    def __init__(self) -> None:
        """Inicializa los componentes del pipeline y maneja la carga segura del modelo SER."""
        self.stt = SpeechToTextWhisper()
        self.ser_error: Optional[str] = None

        try:
            self.ser = EmotionSER()
        except FileNotFoundError as exc:
            # No se interrumpe la carga de la interfaz; el motivo se conserva para informarlo en ejecución.
            self.ser = None
            self.ser_error = str(exc)
        except Exception as exc:
            self.ser = None
            self.ser_error = f"Fallo inesperado al cargar SER: {exc}"

        # TTS (pyttsx3) se encapsula con bloqueo y tiempo de espera para mitigar bloqueos intermitentes.
        self._tts_lock = threading.Lock()
        self._tts_blocked = False
        self._tts_thread: Optional[threading.Thread] = None

        self.chat_responder = chat_responder

    def _build_tts_engine(self):
        engine = pyttsx3.init()
        if settings.tts_voice is not None:
            target = settings.tts_voice.lower()
            try:
                for voice in engine.getProperty("voices"):
                    name = getattr(voice, "name", "") or ""
                    if target in name.lower():
                        engine.setProperty("voice", voice.id)
                        break
            except Exception:
                # En caso de fallo en el listado de voces, se mantiene la voz por defecto.
                pass
        return engine

    def _synthesize_tts(self, text: str) -> Optional[str]:
        if not getattr(settings, "enable_tts", False):
            return None
        if self._tts_blocked and self._tts_thread and not self._tts_thread.is_alive():
            self._tts_blocked = False
            self._tts_thread = None
        if self._tts_blocked:
            return None

        tmp_dir = Path(tempfile.gettempdir())
        unique_name = f"emotion_assistant_response_{uuid.uuid4().hex}.wav"
        out_path = tmp_dir / unique_name

        acquired = self._tts_lock.acquire(timeout=TTS_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            # Se evitan interbloqueos si un motor anterior quedó bloqueado.
            return None
        if self._tts_blocked:
            self._tts_lock.release()
            return None

        errors: list[Exception] = []

        def _worker():
            engine = None
            try:
                engine = self._build_tts_engine()
                engine.save_to_file(text, str(out_path))
                engine.runAndWait()
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass

        try:
            tts_thread = threading.Thread(target=_worker, daemon=True)
            self._tts_thread = tts_thread
            tts_thread.start()
            tts_thread.join(timeout=TTS_TIMEOUT_SECONDS)

            if tts_thread.is_alive():
                # Se marca el TTS como bloqueado para evitar reintentos mientras persista el bloqueo.
                self._tts_blocked = True
                try:
                    if out_path.exists():
                        out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None

            if errors:
                try:
                    if out_path.exists():
                        out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None

            return str(out_path) if out_path.exists() else None
        finally:
            if self._tts_lock.locked():
                self._tts_lock.release()

    def synthesize_text(self, text: str) -> Optional[str]:
        return self._synthesize_tts(text)

    def process(
        self,
        audio_path: str | Path,
        use_ollama: bool = False,
        use_tts: bool = False,
    ) -> PipelineResult:
        """Ejecuta el flujo completo sobre un audio (sin historial)."""
        if self.ser is None:
            raise RuntimeError(self.ser_error or "Modelo SER no disponible.")

        t0 = time.perf_counter()

        # 1) STT
        stt_result = self.stt.transcribe(audio_path)

        # 2) SER
        ser_pred: EmotionPrediction = self.ser.predict(audio_path)
        t2 = time.perf_counter()

        # 3) Respuesta (plantilla/parafraseo con LLM)
        resp: ResponseResult = generate_response(
            emotion=ser_pred.label,
            use_ollama=use_ollama,
            transcript=stt_result.get("text"),
            emotion_score=ser_pred.score,
        )
        t3 = time.perf_counter()

        # 4) TTS
        response_audio_path: Optional[str] = None
        tts_latency = 0.0
        if use_tts and settings.enable_tts:
            response_audio_path = self._synthesize_tts(resp.final_text)
            tts_latency = time.perf_counter() - t3

        t_end = time.perf_counter()

        latencies = {
            "stt": stt_result["latency"],
            "ser": ser_pred.latency,
            "response": t3 - t2,
            "tts": tts_latency,
            "total": t_end - t0,
        }

        return PipelineResult(
            transcript=stt_result["text"],
            emotion=ser_pred.label,
            emotion_score=ser_pred.score,
            response_text=resp.final_text,
            response_audio_path=response_audio_path,
            latencies=latencies,
            scores_per_class=ser_pred.scores_per_class,
            used_ollama=resp.used_ollama,
        )

    def process_turn(
        self,
        audio_path: str | Path,
        history: Optional[List[Dict[str, str]]] = None,
        auto_tts: bool = False,
    ) -> ChatPipelineResult:
        """
        Procesa un turno conversacional completo:
            Entrada (audio) -> STT -> SER -> LLM -> TTS -> Respuesta (texto/audio)
        """
        if self.ser is None:
            raise RuntimeError(self.ser_error or "Modelo SER no disponible.")

        t0 = time.perf_counter()

        # 1) STT
        stt_result = self.stt.transcribe(audio_path)
        transcript_text = (stt_result.get("text") or "").strip()

        # 2) SER
        ser_pred: EmotionPrediction = self.ser.predict(audio_path)
        t2 = time.perf_counter()

        # 3) Respuesta conversacional
        llm_history = history or []
        # Se considera inválida la transcripción si no contiene caracteres alfanuméricos.
        stt_invalid = not any(ch.isalnum() for ch in transcript_text)
        if stt_invalid:
            assistant_text = "No he podido entenderte bien, ¿puedes repetirlo?"
            used_ollama = False
            user_for_history = transcript_text if transcript_text else "(audio no entendido)"
            updated_history = self.chat_responder._append_turn(
                llm_history,
                user_for_history,
                assistant_text,
            )
            t3 = t2  # sin llamada al LLM.
        else:
            chat_resp: ChatResponseResult = self.chat_responder.chat(
                transcript=transcript_text,
                emotion=ser_pred.label,
                emotion_score=ser_pred.score,
                history=llm_history,
            )
            assistant_text = chat_resp.assistant_text
            used_ollama = chat_resp.used_ollama
            updated_history = chat_resp.history
            t3 = time.perf_counter()

        # 4) TTS
        response_audio_path: Optional[str] = None
        tts_latency = 0.0
        if auto_tts and settings.enable_tts:
            response_audio_path = self._synthesize_tts(assistant_text)
            tts_latency = time.perf_counter() - t3

        t_end = time.perf_counter()

        latencies = {
            "stt": stt_result["latency"],
            "ser": ser_pred.latency,
            "response": t3 - t2,
            "tts": tts_latency,
            "total": t_end - t0,
        }

        return ChatPipelineResult(
            transcript=transcript_text,
            emotion=ser_pred.label,
            emotion_score=ser_pred.score,
            assistant_text=assistant_text,
            response_audio_path=response_audio_path,
            latencies=latencies,
            scores_per_class=ser_pred.scores_per_class,
            used_ollama=used_ollama,
            history=updated_history,
        )
