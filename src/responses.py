"""Generación de respuestas del asistente.

Incluye:
- Plantillas controladas como mecanismo de reserva.
- Parafraseo mediante LLM local.
- Modo conversacional con historial acotado a través de Ollama.

El objetivo es mantener respuestas breves, seguras y coherentes con el contexto
conversacional, preservando la trazabilidad en los experimentos.
"""

from __future__ import annotations

import json
import math
import unicodedata
import textwrap
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from .config import settings

###############################################################################
# Plantillas controladas (respaldo)
###############################################################################

# Estas plantillas se utilizan como mecanismo de reserva cuando no hay LLM o cuando
# se decide responder de forma prudente (p. ej., baja confianza del SER).
# Se diseñan con los siguientes criterios:
# - breves, claras y no clínicas
# - accionables (micro-intervenciones de 30–90 s)
# - respetuosas con la autonomía del usuario ("si te apetece", "si te sirve")
#
# Nota: se proporcionan variantes por emoción para reducir la monotonía.


TEMPLATE_VARIANTS: Dict[str, List[str]] = {
    "alegria": [
        (
            "Se nota que estás viviendo algo positivo. Si te apetece, tómate 20–30 segundos para saborearlo: "
            "observa qué cambia en tu cuerpo (respiración, postura, tensión) y qué detalle lo hizo posible. "
            "¿Qué te gustaría hacer ahora para prolongar esta sensación de forma sencilla?"
        ),
        (
            "Me alegra leer eso. Una forma de consolidarlo es ponerle nombre: "
            "¿qué emoción concreta es (alivio, orgullo, ilusión) y qué te dice sobre lo que valoras? "
            "Si te sirve, anota una cosa que quieras repetir mañana o esta semana."
        ),
    ],
    "ira": [
        (
            "Lo que describes suena muy frustrante. Si te ayuda, prueba una pausa corta antes de responder: "
            "afloja mandíbula y hombros, y haz 3 respiraciones con la exhalación un poco más larga que la inhalación. "
            "Después pregúntate: ¿qué necesito ahora mismo (límite, claridad, descanso) y cuál sería un siguiente paso pequeño?"
        ),
        (
            "Entiendo el enfado. A veces baja la intensidad si se separa el hecho de la interpretación: "
            "¿qué pasó exactamente (en una frase) y qué parte depende de ti? "
            "Si te apetece, elige una acción concreta y breve (p. ej., pedir un minuto, escribir lo que quieres decir y revisarlo)."
        ),
    ],
    "tristeza": [
        (
            "Siento que estés pasando por esto. Si te sirve, date permiso para sentirlo sin pelearte con ello un momento. "
            "Luego prueba una acción muy pequeña de cuidado (agua, ducha, abrir la ventana, ordenar una cosa). "
            "¿Qué sería lo más amable contigo que podrías hacer en los próximos 10 minutos?"
        ),
        (
            "Debe ser pesado sostener esa tristeza. Una técnica breve es volver a lo concreto: "
            "mira a tu alrededor y nombra 5 cosas que ves, 4 que sientes al tacto y 3 sonidos que escuchas. "
            "Si quieres, cuéntame qué ha sido lo más difícil de hoy (con una frase)."
        ),
    ],
    "miedo": [
        (
            "Es normal sentir inquietud cuando algo se percibe incierto. Si te apetece, prueba un anclaje rápido: "
            "apoya bien los pies, nota el contacto con el suelo y haz 5–10 segundos de exhalación lenta (sin forzar). "
            "Si centrarte en la respiración te incomoda, vuelve al entorno con la regla de 5-4-3-2-1 (ver, tocar, oír, oler, saborear)."
        ),
        (
            "Cuando aparece el miedo, el cuerpo intenta protegerte. Puede ayudar distinguir "
            "entre “peligro real” y “posibilidad temida”: ¿qué evidencia tienes ahora mismo de que está ocurriendo lo peor? "
            "Si te sirve, elige una acción de control mínimo (p. ej., recopilar una información, pedir apoyo, o planear el primer paso)."
        ),
    ],
    "neutro": [
        (
            "Gracias por compartirlo. Si ahora todo está más plano, puede ser un buen momento para un chequeo rápido: "
            "¿cómo está tu energía (0–10) y dónde notas más tensión? "
            "Si te apetece, haz un estiramiento suave de cuello/hombros y elige una prioridad pequeña para el siguiente rato."
        ),
        (
            "Entendido. A veces lo “neutro” es solo falta de señal clara. "
            "Si te sirve, pon nombre a lo que necesitas: ¿pausa, foco, compañía o movimiento? "
            "Puedo proponerte una técnica breve según lo que busques."
        ),
    ],
}

TEMPLATE_VARIANTS_LOW_CONFIDENCE: List[str] = [
    (
        "Gracias por compartirlo. Si te sirve, tomemos una pausa breve: "
        "nota cómo está tu respiración y relaja hombros y mandíbula. "
        "Podemos ir paso a paso y enfocarnos en lo que necesitas ahora mismo."
    ),
    (
        "Estoy aquí contigo. A veces ayuda volver a lo básico: "
        "respira suave 3 veces y mira a tu alrededor para anclarte al presente. "
        "Si quieres, añade un detalle sobre lo que está pasando."
    ),
    (
        "Gracias por contarlo. Podemos ir despacio: "
        "elige una acción pequeña y concreta para este momento "
        "(por ejemplo, agua, estirarte o salir al aire un minuto)."
    ),
    (
        "Te escucho. Si ahora todo está mezclado, probemos algo sencillo: "
        "apoya bien los pies y haz una exhalación lenta. "
        "Después podemos continuar con más calma."
    ),
]

TEMPLATE_CRISIS: str = (
    "Siento que estés pasando por algo así. Si estás en peligro inmediato o temes hacerte daño, "
    "por favor, contacta con los servicios de emergencia de tu zona (por ejemplo, 112 en la UE) o con alguien de confianza ahora mismo. "
    "Si puedes, dime dónde estás (país) y si estás a salvo en este momento; puedo ayudarte a buscar el recurso adecuado."
)


CRISIS_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"\b(suicid|quitarme la vida|me quiero morir|no quiero vivir)\b", re.IGNORECASE),
    re.compile(r"\b(hacerme daño|autolesion|autolesi[oó]n|cortarme)\b", re.IGNORECASE),
    re.compile(r"\b(matarme|matar a alguien|hacer daño a alguien)\b", re.IGNORECASE),
]


def _looks_like_crisis(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in CRISIS_PATTERNS)


def _is_low_confidence(score: Optional[float]) -> bool:
    return score is not None and score < settings.emotion_low_confidence_threshold


def _normalize_emotion_key(emotion: str) -> str:
    """Normaliza etiquetas para que coincidan con TEMPLATE_VARIANTS."""
    value = (emotion or "").strip().lower()
    if not value:
        return "neutro"

    if value.isdigit():
        try:
            value = settings.ser_id2label.get(int(value), value)
        except Exception:
            pass

    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))

    aliases = {
        "happiness": "alegria",
        "joy": "alegria",
        "anger": "ira",
        "sadness": "tristeza",
        "fear": "miedo",
        "neutral": "neutro",
        "disgust": "neutro",
    }
    value = aliases.get(value, value)

    if value not in TEMPLATE_VARIANTS:
        return "neutro"

    return value


def _stable_choice_index(key: str, n: int) -> int:
    """Selección estable (sin aleatoriedad) a partir de una clave."""
    if n <= 1:
        return 0
    digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % n


def build_template(
    emotion: str,
    emotion_score: Optional[float] = None,
    transcript: Optional[str] = None,
) -> Tuple[str, str]:
    """Devuelve (template_text, template_kind).

    template_kind se expone para trazabilidad (p. ej., "emotion", "low_confidence", "crisis").
    """
    if transcript and _looks_like_crisis(transcript):
        return TEMPLATE_CRISIS, "crisis"

    if _is_low_confidence(emotion_score):
        variants = TEMPLATE_VARIANTS_LOW_CONFIDENCE
        selector = f"low_confidence|{(transcript or '').strip()}"
        idx = _stable_choice_index(selector, len(variants))
        return variants[idx], "low_confidence"

    key = _normalize_emotion_key(emotion)
    variants = TEMPLATE_VARIANTS.get(key) or TEMPLATE_VARIANTS["neutro"]

    # Se aplica una elección estable para garantizar la reproducibilidad experimental.
    selector = f"{key}|{(transcript or '').strip()}"
    idx = _stable_choice_index(selector, len(variants))
    return variants[idx], "emotion"


# Compatibilidad: se mantiene un diccionario simple con la variante "principal" por emoción.
TEMPLATES: Dict[str, str] = {
    k: v[0] for k, v in TEMPLATE_VARIANTS.items()
}


@dataclass
class ResponseResult:
    template_text: str
    final_text: str
    used_ollama: bool


@dataclass
class ChatResponseResult:
    assistant_text: str
    history: List[Dict[str, str]]
    used_ollama: bool
    template_text: str


class SafeParaphraser:
    """Parafraseo con LLM local (Ollama) bajo reglas básicas de seguridad."""
    FORBIDDEN: List[str] = [
        "diagnóstico",
        "diagnostico",
        "recetar",
        "medicación",
        "medicacion",
        "pastillas",
        "fármaco",
        "farmaco",
    ]

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def _build_prompt(self, text: str) -> str:
        return textwrap.dedent(
            f"""
            Eres un asistente empático y cercano. Tu tarea es parafrasear el siguiente texto
            para que suene natural, cálido y humano en español.

            Reglas:
            - Mantén el mensaje de apoyo, pero usa tus propias palabras.
            - NO añadas diagnósticos médicos ni recomiendes medicación.
            - Sé breve pero no cortante (máximo de {settings.ollama_max_tokens} tokens).
            - Evita frases robóticas como "Es comprensible que...". Habla como un compañero.

            Texto a parafrasear:
            \"\"\"{text}\"\"\"
            """
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": settings.ollama_temperature,
                "top_p": settings.ollama_top_p,
                "num_predict": settings.ollama_max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()

        text_chunks: List[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # El endpoint devuelve la salida en streaming; se concatena cada fragmento recibido.
            data = json.loads(line)
            if "response" in data:
                text_chunks.append(data["response"])
        return "".join(text_chunks).strip()

    def _passes_checks(self, text: str) -> bool:
        # Se controla la longitud en palabras para evitar desvíos extensos.
        words = text.split()
        if len(words) < 10 or len(words) > 80:
            return False

        lowered = text.lower()
        return not any(bad in lowered for bad in self.FORBIDDEN)

    def paraphrase(self, text: str) -> str | None:
        try:
            prompt = self._build_prompt(text)
            candidate = self._call_ollama(prompt)
        except Exception:
            return None

        if self._passes_checks(candidate):
            return candidate
        return None


paraphraser = SafeParaphraser()


class ChatResponder:
    """Respuestas conversacionales mediante /api/chat (Ollama).

    Se utiliza un historial acotado y la emoción detectada como contexto implícito.
    """

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.max_turns = max(1, int(settings.max_history_turns))
        self.enabled = settings.use_ollama

        # Se reutiliza la lista de términos “sensibles” del parafraseador para filtrar
        # salidas del modo chat cuando se observa desvío hacia consejos médicos.
        self._forbidden = set(SafeParaphraser.FORBIDDEN)

    def _system_prompt(self) -> str:
        return textwrap.dedent(
            """
            Eres un asistente experimental de apoyo emocional en español.

            Objetivo:
            - Ofrecer una respuesta breve, prudente y útil (2–6 frases cortas).
            - Incluir como máximo UNA micro-técnica (30–90 s) de corte general (respiración suave, anclaje al entorno,
              relajación muscular breve, o reformulación simple), sin lenguaje clínico.

            Reglas:
            - No hagas diagnósticos, no prescribas tratamientos ni recomiendes medicación.
            - Evita afirmaciones absolutas; ofrece opciones (“si te sirve…”, “si te apetece…”).
            - No inventes datos personales ni asumas detalles no mencionados.
            - No repitas literalmente la entrada del usuario.
            - Ajusta el tono a la emoción detectada, pero NO menciones la etiqueta emocional salvo que el usuario lo pida.
            - Si el usuario expresa riesgo de hacerse daño o urgencia, prioriza seguridad y sugiere buscar ayuda inmediata.
            """
        ).strip()

    def _trim_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Limita el historial a los últimos N turnos.
        Cada turno aporta dos mensajes, por lo que se recorta a 2 * max_turns.
        """
        cleaned: List[Dict[str, str]] = []
        for msg in history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant" and not cleaned:
                continue  # Se evita iniciar el historial con un mensaje del asistente.
            if not content:
                continue
            cleaned.append({"role": role, "content": content})

        max_messages = self.max_turns * 2
        if len(cleaned) <= max_messages:
            return cleaned
        return cleaned[-max_messages:]

    def _build_messages(
        self,
        transcript: str,
        emotion: str,
        emotion_score: Optional[float],
        history: List[Dict[str, str]],
        base_template: str,
        low_confidence: bool,
    ) -> List[Dict[str, str]]:
        trimmed_history = self._trim_history(history)

        if low_confidence:
            context_note = (
                "Contexto interno: la emoción detectada tiene confianza baja. "
                "No asumas una emoción concreta; prioriza un tono neutro y útil, "
                "y evita mencionar la etiqueta salvo que el usuario lo pida. "
            )
            context_note += (
                "Si realmente falta contexto para ayudar, formula UNA pregunta "
                "de clarificación breve; si no, evita preguntar."
            )
        else:
            score_note = "confianza desconocida"
            if emotion_score is not None and math.isfinite(emotion_score):
                score_note = f"confianza {emotion_score:.2f}"
            context_note = (
                f"Contexto interno: emoción predominante detectada '{emotion}' "
                f"con {score_note}. Úsalo solo para adaptar el tono, "
                "no lo menciones salvo que el usuario lo pida explícitamente."
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(trimmed_history)
        messages.append({"role": "system", "content": context_note})
        messages.append(
            {
                "role": "system",
                "content": (
                    "Guía de respuesta (usa esto como base si necesitas estructura): "
                    f"{base_template}"
                ),
            }
        )
        messages.append({"role": "user", "content": transcript})
        return messages

    def _passes_llm_checks(self, text: str) -> bool:
        """Filtros ligeros para mantener la salida dentro de límites prudentes."""
        t = (text or "").strip()
        if not t:
            return False

        # Se evitan respuestas excesivas o demasiado cortas.
        words = t.split()
        if len(words) < 8 or len(words) > 140:
            return False

        lowered = t.lower()
        if any(bad in lowered for bad in self._forbidden):
            return False

        return True

    def _append_turn(
        self,
        history: List[Dict[str, str]],
        user_text: str,
        assistant_text: str,
    ) -> List[Dict[str, str]]:
        updated = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        return self._trim_history(updated)

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": settings.ollama_temperature,
                "top_p": settings.ollama_top_p,
                "num_predict": settings.ollama_max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()

        text_chunks: List[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if "message" in data and isinstance(data["message"], dict):
                content = data["message"].get("content") or ""
                text_chunks.append(content)
            elif "response" in data:
                text_chunks.append(data["response"])

        assistant_text = "".join(text_chunks).strip()
        if not assistant_text:
            raise ValueError("Respuesta vacía del LLM.")
        return assistant_text

    def chat(
        self,
        transcript: str,
        emotion: str,
        emotion_score: float,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResponseResult:
        history = history or []

        base_template, template_kind = build_template(
            emotion=emotion,
            emotion_score=emotion_score,
            transcript=transcript,
        )

        # Se prioriza una respuesta segura ante riesgo o urgencia.
        if template_kind == "crisis":
            updated_history = self._append_turn(history, transcript, base_template)
            return ChatResponseResult(
                assistant_text=base_template,
                history=updated_history,
                used_ollama=False,
                template_text=base_template,
            )

        if not self.enabled:
            updated_history = self._append_turn(history, transcript, base_template)
            return ChatResponseResult(
                assistant_text=base_template,
                history=updated_history,
                used_ollama=False,
                template_text=base_template,
            )

        try:
            messages = self._build_messages(
                transcript=transcript,
                emotion=emotion,
                emotion_score=emotion_score,
                history=history,
                base_template=base_template,
                low_confidence=(template_kind == "low_confidence"),
            )
            candidate = self._call_ollama(messages)
            if self._passes_llm_checks(candidate):
                assistant_text = candidate
                used_ollama = True
            else:
                assistant_text = base_template
                used_ollama = False
        except Exception:
            assistant_text = base_template
            used_ollama = False

        updated_history = self._append_turn(history, transcript, assistant_text)
        return ChatResponseResult(
            assistant_text=assistant_text,
            history=updated_history,
            used_ollama=used_ollama,
            template_text=base_template,
        )


chat_responder = ChatResponder()


def generate_response(
    emotion: str,
    use_ollama: bool,
    transcript: Optional[str] = None,
    emotion_score: Optional[float] = None,
) -> ResponseResult:
    base, _ = build_template(
        emotion=emotion,
        emotion_score=emotion_score,
        transcript=transcript,
    )

    if not use_ollama or not settings.use_ollama:
        return ResponseResult(template_text=base, final_text=base, used_ollama=False)

    candidate = paraphraser.paraphrase(base)
    if candidate is None:
        return ResponseResult(template_text=base, final_text=base, used_ollama=False)

    return ResponseResult(template_text=base, final_text=candidate, used_ollama=True)
