"""Interfaz de usuario (Gradio) del MVP.

Se presenta un chat con entrada por audio y paneles auxiliares para:
- Última respuesta en audio (TTS).
- Emoción detectada y métricas (latencias y probabilidades).
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf

from .config import settings
from .pipeline import EmotionAssistantPipeline


pipeline = EmotionAssistantPipeline()

GradioAudio = tuple[int, np.ndarray]


def _safe_remove(path: str | None) -> None:
    """Elimina un fichero si existe, ignorando errores de E/S."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        # Un fallo en el borrado no afecta al funcionamiento del MVP.
        pass


def _audio_duration_seconds(audio: GradioAudio) -> float:
    sr, y = audio
    if sr <= 0:
        return 0.0
    return float(len(y)) / float(sr)


def _to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return y.mean(axis=1)
    return y


def _prepare_audio_input(
    audio_in: GradioAudio | str | None,
    max_duration_s: float | None = None,
    min_duration_s: float | None = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Normaliza la entrada de audio y devuelve: audio_path, tmp_in_wav y error_msg."""
    if audio_in is None:
        return None, None, "No se ha recibido audio."

    tmp_in_wav: Optional[str] = None

    if max_duration_s is None:
        max_duration_s = float(getattr(settings, "audio_max_duration_s", 20.0))
    if min_duration_s is None:
        min_duration_s = float(getattr(settings, "audio_min_duration_s", 0.0))
    if max_duration_s < 0:
        max_duration_s = 0.0
    if min_duration_s < 0:
        min_duration_s = 0.0
    if max_duration_s < min_duration_s:
        max_duration_s = min_duration_s

    if isinstance(audio_in, tuple):
        sr, y = audio_in
        y = _to_mono(np.asarray(y))

        duration_s = _audio_duration_seconds((int(sr), y))
        if min_duration_s > 0 and duration_s < min_duration_s:
            return None, None, (
                f"Audio demasiado corto ({duration_s:.2f} s). "
                f"Graba al menos {min_duration_s:.1f} s."
            )
        if duration_s > max_duration_s:
            return None, None, (
                f"Audio demasiado largo ({duration_s:.1f} s). "
                f"Vuelve a grabar con una duración máxima de {int(max_duration_s)} s."
            )

        f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_in_wav = f.name
        f.close()
        sf.write(tmp_in_wav, y, int(sr), subtype="PCM_16")
        audio_path = tmp_in_wav
    else:
        audio_path = audio_in
        if not os.path.exists(audio_path):
            return None, None, f"No existe el archivo de audio: {audio_path}"
        try:
            with sf.SoundFile(audio_path) as f:
                duration_s = len(f) / float(f.samplerate)
            if min_duration_s > 0 and duration_s < min_duration_s:
                return None, None, (
                    f"Audio demasiado corto ({duration_s:.2f} s). "
                    f"Graba al menos {min_duration_s:.1f} s."
                )
            if duration_s > max_duration_s:
                return None, None, (
                    f"Audio demasiado largo ({duration_s:.1f} s). "
                    f"Vuelve a grabar con una duración máxima de {int(max_duration_s)} s."
                )
        except Exception:
            return None, None, "No se pudo leer el audio. Verifica el formato o la ruta."

    return audio_path, tmp_in_wav, None

def _pairs_from_llm_history(llm_history: List[dict] | None) -> List[Tuple[str, str]]:
    """Convierte mensajes (system/user/assistant) en pares (user, assistant) para el chatbot."""
    chat: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None

    for msg in llm_history or []:
        role = (msg or {}).get("role", "")
        content = (msg or {}).get("content", "") or ""
        if role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            chat.append((pending_user, content))
            pending_user = None

    return chat


def _trim_llm_history(llm_history: List[dict], max_turns: int) -> List[dict]:
    """Mantiene el prefijo inicial 'system' y los últimos 'max_turns' turnos."""
    if not llm_history:
        return []

    prefix_end = 0
    while prefix_end < len(llm_history) and (llm_history[prefix_end] or {}).get("role") == "system":
        prefix_end += 1

    user_idxs = [i for i, m in enumerate(llm_history) if (m or {}).get("role") == "user"]
    if len(user_idxs) <= max_turns:
        return llm_history

    start_user = user_idxs[-max_turns]
    start = start_user

    while start > prefix_end and (llm_history[start - 1] or {}).get("role") == "system":
        start -= 1

    return llm_history[:prefix_end] + llm_history[start:]


def _format_metrics(latencies: dict, scores_per_class: dict, low_confidence: bool) -> str:
    lines = [
        f"Latencia STT: {latencies.get('stt', 0.0):.3f} s",
        f"Latencia SER: {latencies.get('ser', 0.0):.3f} s",
        f"Latencia de respuesta: {latencies.get('response', 0.0):.3f} s",
        f"Latencia TTS: {latencies.get('tts', 0.0):.3f} s",
        f"Latencia total (E2E): {latencies.get('total', 0.0):.3f} s",
        "",
        "Probabilidades por emoción:",
    ]
    for label, score in sorted(scores_per_class.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {label}: {score:.2f}")

    if low_confidence:
        lines.append("")
        lines.append("⚠️ Nota: confianza baja en la emoción detectada; interpreta el resultado con cautela.")

    return "\n".join(lines)


def process_turn_ui(
    audio_in: GradioAudio | str | None,
    llm_state: List[dict] | None,
):
    """Salidas: chatbot, llm_state, emotion_out, metrics_out, audio_out, audio_input (según keep_audio_after_send)."""

    llm_state = llm_state or []
    keep_audio = bool(getattr(settings, "keep_audio_after_send", False))
    audio_return = audio_in if keep_audio else None

    if getattr(pipeline, "ser_error", None):
        err = f"Error al cargar el modelo de emociones: {pipeline.ser_error}"
        return [], llm_state, "", err, None, audio_return

    tmp_in_wav: Optional[str] = None
    tmp_out_wav: Optional[str] = None

    try:
        audio_path, tmp_in_wav, err = _prepare_audio_input(audio_in)
        if err or audio_path is None:
            chat_pairs = _pairs_from_llm_history(llm_state)
            return chat_pairs, llm_state, "", (err or "Entrada no válida."), None, audio_return

        if not hasattr(pipeline, "process_turn"):
            chat_pairs = _pairs_from_llm_history(llm_state)
            return (
                chat_pairs,
                llm_state,
                "",
                "No se encuentra el método process_turn() en el pipeline.",
                None,
                audio_return,
            )

        result = pipeline.process_turn(
            audio_path=audio_path,
            history=llm_state,
            auto_tts=bool(getattr(settings, "enable_tts", False)),
        )

        new_llm = _trim_llm_history(result.history, settings.max_history_turns)
        pairs = _pairs_from_llm_history(new_llm)
        if len(pairs) > settings.max_history_turns:
            pairs = pairs[-settings.max_history_turns :]

        low_conf = (
            (result.emotion_score is not None)
            and (result.emotion_score < settings.emotion_low_confidence_threshold)
        )
        emotion_str = (
            f"{result.emotion} (confianza baja: {result.emotion_score:.2f})"
            if low_conf
            else f"{result.emotion} (confianza: {result.emotion_score:.2f})"
        )

        metrics_text = _format_metrics(result.latencies, result.scores_per_class, low_conf)

        audio_out = None
        if getattr(result, "response_audio_path", None):
            tmp_out_wav = result.response_audio_path
            # Se normaliza la salida a mono y se limita el rango para su reproducción en la interfaz.
            try:
                data, sr = sf.read(tmp_out_wav, dtype="float32", always_2d=False)
                data = _to_mono(np.asarray(data)).astype(np.float32, copy=False)
                data = np.clip(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                audio_out = (int(sr), data)
            except Exception:
                audio_out = None

        # Se limpia o se conserva el audio original según settings.keep_audio_after_send.
        return pairs, new_llm, emotion_str, metrics_text, audio_out, audio_return

    except Exception as exc:
        chat_pairs = _pairs_from_llm_history(llm_state)
        return chat_pairs, llm_state, "", f"Error al procesar el audio: {exc}", None, audio_return

    finally:
        _safe_remove(tmp_in_wav)
        _safe_remove(tmp_out_wav)


def clear_all():
    """Restablece el estado de la interfaz (chat, estados, emoción, métricas y audios)."""
    return [], [], "", "", None, None


def build_demo():
    header_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    :root {
        --hero-accent: #ff7a18;
        --hero-accent-2: #ffb347;
        --hero-text: #f8f9fb;
        --hero-muted: #cbd3dc;
        --hero-border: rgba(255, 255, 255, 0.08);
        --surface-0: #0b0c0f;
        --surface-1: #151820;
        --surface-2: #1d222c;
        --surface-3: #232935;
        --surface-border: rgba(255, 255, 255, 0.08);
        --text-primary: #f6f7fb;
        --text-muted: #b9c2cc;
        --accent: #ff7a18;
        --accent-strong: #ff9a2f;
        --accent-soft: rgba(255, 122, 24, 0.18);
        --hero-bg: radial-gradient(120% 140% at 8% 0%, rgba(255, 122, 24, 0.20), transparent 60%),
            radial-gradient(140% 160% at 90% -10%, rgba(255, 179, 71, 0.20), transparent 55%),
            linear-gradient(135deg, rgba(15, 16, 20, 0.92) 0%, rgba(21, 23, 30, 0.96) 100%);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 32px;
        margin-bottom: 18px;
        border-radius: 18px;
        border: 1px solid var(--hero-border);
        background: var(--hero-bg);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
        animation: heroFade 720ms ease-out;
    }

    .hero__bg {
        position: absolute;
        inset: 0;
        background:
            linear-gradient(120deg, rgba(255, 255, 255, 0.04), transparent 40%),
            radial-gradient(60% 60% at 60% 40%, rgba(255, 255, 255, 0.05), transparent 65%);
        opacity: 0.85;
        pointer-events: none;
    }

    .hero__content {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-family: "Work Sans", "Segoe UI", sans-serif;
        color: var(--hero-text);
    }

    .hero__eyebrow {
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        font-size: 12px;
        letter-spacing: 0.08em;
        color: var(--hero-accent-2);
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .hero__eyebrow-left {
        font-weight: 700;
        letter-spacing: 0.06em;
        color: var(--hero-text);
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: none;
        cursor: default;
    }

    .hero__eyebrow-mid {
        font-weight: 600;
        letter-spacing: 0.08em;
        color: var(--hero-accent-2);
    }

    .hero__title {
        margin: 0;
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        font-weight: 700;
        line-height: 1.1;
        font-size: clamp(26px, 3.1vw, 40px);
    }

    .hero__lead {
        margin: 0;
        color: var(--hero-muted);
        font-size: 15px;
        line-height: 1.6;
        max-width: 96ch;
    }

    .hero__meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }

    .hero__meta-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--hero-muted);
    }

    .hero__pill {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.01em;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #f8f0e8;
        font-weight: 650;
        cursor: default;
        box-shadow: none;
    }

    .hero__notice {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 10px 12px 12px;
    }

    .hero__notice-head {
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }

    .hero__notice-title {
        font-weight: 700;
        color: #ffd9b0;
        white-space: nowrap;
    }

    .hero__notice-text {
        color: var(--hero-muted);
        font-size: 14px;
        line-height: 1.5;
    }

    .hero__notice-body {
        padding-top: 6px;
        color: var(--hero-muted);
        font-size: 13.5px;
        line-height: 1.5;
    }

    .hero__notice-body ul {
        margin: 0;
        padding-left: 18px;
    }

    .hero__content > * {
        animation: heroRise 720ms ease-out both;
    }

    .hero__content > *:nth-child(1) { animation-delay: 80ms; }
    .hero__content > *:nth-child(2) { animation-delay: 140ms; }
    .hero__content > *:nth-child(3) { animation-delay: 200ms; }
    .hero__content > *:nth-child(4) { animation-delay: 260ms; }
    .hero__content > *:nth-child(5) { animation-delay: 320ms; }

    @keyframes heroFade {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes heroRise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 720px) {
        .hero { padding: 22px 20px; }
        .hero__title { font-size: clamp(22px, 6vw, 32px); }
        .hero__lead { font-size: 14px; }
        .hero__notice-head { flex-direction: column; }
    }

    @media (prefers-reduced-motion: reduce) {
        .hero,
        .hero__content > * {
            animation: none;
        }
    }

    .gradio-container {
        background:
            radial-gradient(100% 160% at 8% -30%, rgba(255, 122, 24, 0.18), transparent 60%),
            radial-gradient(100% 120% at 92% 0%, rgba(255, 179, 71, 0.12), transparent 55%),
            linear-gradient(180deg, #0a0b0f 0%, #0f1117 100%);
        color: var(--text-primary);
        font-family: "Work Sans", "Segoe UI", sans-serif;
    }

    #hero-wrap {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
    }

    #layout-row {
        gap: 18px;
    }

    #col-main,
    #col-side {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    #clear-btn {
        margin-top: auto;
    }

    #chat-panel,
    #audio-input,
    #audio-out,
    #emotion-out,
    #metrics-out {
        background: var(--surface-1);
        border: 1px solid var(--surface-border);
        border-radius: 16px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }

    .gradio-container label,
    .gradio-container .label {
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        letter-spacing: 0.04em;
        font-size: 12px;
        color: #d5dde7;
    }

    .gradio-container textarea,
    .gradio-container input[type="text"] {
        background: var(--surface-2);
        border: 1px solid var(--surface-border);
        color: var(--text-primary);
        border-radius: 12px;
    }

    .gradio-container textarea::placeholder,
    .gradio-container input[type="text"]::placeholder {
        color: rgba(213, 221, 231, 0.6);
    }

    #send-btn button {
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        border: 1px solid rgba(255, 175, 100, 0.4);
        color: #1b1106;
        border-radius: 14px;
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        letter-spacing: 0.02em;
        font-weight: 700;
        box-shadow: 0 16px 40px rgba(255, 122, 24, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.35);
        transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
    }

    #send-btn button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
    }

    #send-btn button:focus-visible {
        outline: 2px solid rgba(255, 154, 47, 0.65);
        outline-offset: 2px;
    }

    #send-btn button:active {
        transform: translateY(1px);
        box-shadow: 0 10px 28px rgba(255, 122, 24, 0.3);
    }

    #clear-btn button {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: var(--text-primary);
        border-radius: 14px;
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        letter-spacing: 0.02em;
        box-shadow: none;
    }

    #clear-btn button:hover {
        background: rgba(255, 255, 255, 0.12);
    }

    #audio-input,
    #audio-out {
        background: var(--surface-2);
    }

    #emotion-out textarea,
    #metrics-out textarea {
        background: var(--surface-2);
        border-radius: 12px;
    }

    @media (max-width: 720px) {
        #layout-row {
            gap: 16px;
        }
    }
    """

    with gr.Blocks(title="UNIR - TFE (MVP)", css=header_css) as demo:
        gr.HTML(
            """
            <section class="hero">
              <div class="hero__bg"></div>
              <div class="hero__content">
                <div class="hero__eyebrow">
                  <span class="hero__eyebrow-left" title="Universidad Internacional de La Rioja (UNIR)" aria-label="Universidad Internacional de La Rioja (UNIR)">UNIR</span>
                  <span class="hero__eyebrow-mid">TFE · Desarrollo Software (MVP)</span>
                </div>
                <h1 class="hero__title">Asistente conversacional de apoyo a la regulación emocional</h1>
                <p class="hero__lead">
                  Analiza la voz en español, detecta la emoción predominante y genera una respuesta breve y prudente.
                </p>
                <div class="hero__meta">
                  <span class="hero__meta-label">Emociones que puede detectar:</span>
                  <span class="hero__pill">alegría</span>
                  <span class="hero__pill">ira</span>
                  <span class="hero__pill">tristeza</span>
                  <span class="hero__pill">miedo</span>
                  <span class="hero__pill">neutral</span>
                </div>
                <div class="hero__notice">
                  <div class="hero__notice-head">
                    <span class="hero__notice-title">Aviso importante</span>
                  </div>
                  <div class="hero__notice-body">
                    <ul>
                      <li>No ofrece un diagnóstico ni sustituye la atención de profesionales de la salud mental.</li>
                      <li>Procesamiento local: no se almacenan grabaciones de audio.</li>
                      <li>Salida generada por IA: puede contener errores.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>
            """,
            elem_id="hero-wrap",
        )

        llm_state = gr.State([])

        with gr.Row(elem_id="layout-row"):
            with gr.Column(scale=2, elem_id="col-main"):
                chatbot_kwargs = dict(
                    label="Diálogo",
                    height=360,
                    bubble_full_width=False,
                    show_copy_button=True,
                )
                try:
                    chatbot = gr.Chatbot(
                        **chatbot_kwargs,
                        show_clear_button=False,
                        elem_id="chat-panel",
                    )
                except TypeError:
                    chatbot = gr.Chatbot(**chatbot_kwargs, elem_id="chat-panel")

                max_len = int(getattr(settings, 'audio_max_duration_s', 20.0))
                if max_len < 1:
                    max_len = 1
                audio_input = gr.Audio(
                    label="Habla o sube un fragmento de voz",
                    sources=["microphone", "upload"],
                    type="numpy",
                    max_length=max_len,
                    show_download_button=True,
                    elem_id="audio-input",
                )

                send_btn = gr.Button("Enviar", variant="primary", elem_id="send-btn")

            with gr.Column(scale=1, elem_id="col-side"):
                audio_out = gr.Audio(
                    label="Última respuesta (audio)",
                    type="numpy",
                    visible=bool(getattr(settings, "enable_tts", False)),
                    elem_id="audio-out",
                )

                emotion_out = gr.Textbox(label="Emoción detectada", lines=2, elem_id="emotion-out")
                metrics_out = gr.Textbox(
                    label="Métricas y probabilidades",
                    lines=10,
                    elem_id="metrics-out",
                )

                clear_btn = gr.Button("🧹 Limpiar todo", variant="secondary", elem_id="clear-btn")

        send_btn.click(
            fn=process_turn_ui,
            inputs=[audio_input, llm_state],
            outputs=[chatbot, llm_state, emotion_out, metrics_out, audio_out, audio_input],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, llm_state, emotion_out, metrics_out, audio_out, audio_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    try:
        demo.queue(concurrency_count=1, max_size=10)
    except TypeError:
        demo.queue(max_size=10)
    demo.launch()
