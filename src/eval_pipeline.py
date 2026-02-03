"""Evaluación de STT y del pipeline completo.

Este módulo proporciona utilidades de evaluación para el MVP:
- STT (Whisper): WER/CER y latencias.
- Pipeline completo: latencias por etapa y registro de salidas.

Se utiliza un CSV de entrada con la columna mínima `audio_path` (y opcionalmente
`ref_text` para el cálculo de WER/CER).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jiwer import cer, wer

from .config import settings
from .pipeline import EmotionAssistantPipeline
from .stt_whisper import SpeechToTextWhisper


SER_LABELS = tuple(settings.ser_label2id.keys())


def _percentiles(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _normalize_emotion_label(label: str) -> str:
    value = (label or "").strip().lower()
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _infer_ref_emotion_from_audio_path(audio_path: Path) -> Optional[str]:
    """
    Intenta inferir la etiqueta (5 clases) desde el nombre del fichero o carpeta.
    Ejemplos: 001_alegria_01.wav, .../ira/xxx.wav
    """
    normalized = _normalize_emotion_label(str(audio_path.with_suffix("")))
    tokens = [t for t in re.split(r"[^a-z0-9]+", normalized) if t]
    allowed = set(SER_LABELS)
    for tok in tokens:
        if tok in allowed:
            return tok
    return None


def _classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    labels = SER_LABELS
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t not in label2idx or p not in label2idx:
            continue
        cm[label2idx[t], label2idx[p]] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = (correct / total) if total else 0.0

    recalls: List[float] = []
    f1s: List[float] = []
    per_class: Dict[str, Dict[str, float]] = {}

    for i, lbl in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[lbl] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
        }
        if support > 0:
            recalls.append(recall)
            f1s.append(f1)

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    return {
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_macro": float(f1_macro),
        "total": total,
        "per_class": per_class,
    }


def evaluate_stt(input_csv: Path, output_csv: Path) -> None:
    """
    Evalúa Whisper sobre un CSV con columnas:
      - audio_path (ruta al wav)
      - ref_text (texto de referencia). Si está vacío, no se calcula WER/CER.
    Guarda un CSV con columnas: file, transcript, ref_text, wer, cer, latency_s.
    """
    df = pd.read_csv(input_csv)
    required_cols = {"audio_path"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas mínimas: {required_cols}")

    stt = SpeechToTextWhisper()
    rows_out: List[Dict[str, object]] = []
    wers: List[float] = []
    cers: List[float] = []
    latencies: List[float] = []

    for _, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        ref_text = str(row["ref_text"]).strip() if "ref_text" in row and not pd.isna(row["ref_text"]) else ""
        if not audio_path.exists():
            print(f"[AVISO] No existe audio: {audio_path}", file=sys.stderr)
            continue

        try:
            result = stt.transcribe(audio_path)
        except Exception as exc:
            print(f"[ERROR] STT falló en {audio_path}: {exc}", file=sys.stderr)
            continue

        hyp = result["text"]
        latency = float(result["latency"])
        latencies.append(latency)

        wer_val = cer_val = None
        if ref_text:
            # Jiwer opera sobre cadenas; se utiliza el texto tal cual (sin normalización adicional).
            wer_val = float(wer(ref_text, hyp))
            cer_val = float(cer(ref_text, hyp))
            wers.append(wer_val)
            cers.append(cer_val)

        rows_out.append(
            {
                "file": str(audio_path),
                "transcript": hyp,
                "ref_text": ref_text,
                "wer": wer_val,
                "cer": cer_val,
                "latency_s": latency,
            }
        )

    out_df = pd.DataFrame(rows_out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n=== STT resumen ({len(rows_out)} muestras) ===")
    if wers:
        print(f"WER medio: {np.mean(wers):.3f}")
        print(f"CER medio: {np.mean(cers):.3f}")
    else:
        print("No se calcularon WER/CER (faltan ref_text).")
    lat_pct = _percentiles(latencies)
    print(f"Latencias STT (s): P50={lat_pct['p50']:.3f}  P90={lat_pct['p90']:.3f}  P99={lat_pct['p99']:.3f}")
    print(f"CSV guardado en: {output_csv}")


def evaluate_full_pipeline(
    input_csv: Path,
    output_csv: Path,
    use_ollama: bool = False,
    use_tts: bool = False,
) -> None:
    """
    Ejecuta el pipeline completo y registra:
      transcript, emoción, scores por clase, respuesta, latencias (stt/ser/resp/tts/total).
    CSV de entrada: columna obligatoria audio_path; columna opcional ref_text para WER/CER.
    La emoción de referencia se puede pasar como ref_emotion o se infiere de audio_path.
    """
    df = pd.read_csv(input_csv)
    if "audio_path" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'audio_path'")

    pipeline = EmotionAssistantPipeline()

    rows_out: List[Dict[str, object]] = []
    lat_stt: List[float] = []
    lat_ser: List[float] = []
    lat_resp: List[float] = []
    lat_tts: List[float] = []
    lat_total: List[float] = []
    wers: List[float] = []
    cers: List[float] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for _, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        ref_text = str(row["ref_text"]).strip() if "ref_text" in row and not pd.isna(row["ref_text"]) else ""
        if not audio_path.exists():
            print(f"[AVISO] No existe audio: {audio_path}", file=sys.stderr)
            continue

        try:
            res = pipeline.process(audio_path, use_ollama=use_ollama, use_tts=use_tts)
        except Exception as exc:
            print(f"[ERROR] El pipeline falló en {audio_path}: {exc}", file=sys.stderr)
            continue

        lat = res.latencies
        lat_stt.append(lat["stt"])
        lat_ser.append(lat["ser"])
        lat_resp.append(lat["response"])
        lat_tts.append(lat.get("tts", 0.0))
        lat_total.append(lat["total"])

        wer_val = cer_val = None
        if ref_text:
            wer_val = float(wer(ref_text, res.transcript))
            cer_val = float(cer(ref_text, res.transcript))
            wers.append(wer_val)
            cers.append(cer_val)

        if "ref_emotion" in row and not pd.isna(row["ref_emotion"]):
            ref_emotion = str(row["ref_emotion"]).strip()
        else:
            ref_emotion = _infer_ref_emotion_from_audio_path(audio_path) or ""

        pred_emotion_norm = _normalize_emotion_label(res.emotion)
        ref_emotion_norm = _normalize_emotion_label(ref_emotion)
        emotion_correct: Optional[bool] = None
        if ref_emotion_norm:
            emotion_correct = pred_emotion_norm == ref_emotion_norm
            y_true.append(ref_emotion_norm)
            y_pred.append(pred_emotion_norm)

        row_out = {
            "file": str(audio_path),
            "transcript": res.transcript,
            "ref_text": ref_text,
            "wer": wer_val,
            "cer": cer_val,
            "ref_emotion": ref_emotion,
            "emotion": res.emotion,
            "emotion_correct": emotion_correct,
            "emotion_score": res.emotion_score,
            "scores_per_class": res.scores_per_class,
            "response_text": res.response_text,
            "used_ollama": res.used_ollama,
            "latency_stt_s": lat["stt"],
            "latency_ser_s": lat["ser"],
            "latency_response_s": lat["response"],
            "latency_tts_s": lat.get("tts", 0.0),
            "latency_total_s": lat["total"],
        }
        row_out["response_audio_path"] = res.response_audio_path if res.response_audio_path else ""
        rows_out.append(row_out)

    out_df = pd.DataFrame(rows_out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n=== Pipeline resumen ({len(rows_out)} muestras) ===")
    if wers:
        print(f"WER medio: {np.mean(wers):.3f}")
        print(f"CER medio: {np.mean(cers):.3f}")
    else:
        print("No se calcularon WER/CER (faltan ref_text).")

    if y_true:
        metrics = _classification_metrics(y_true, y_pred)
        print(
            "Resumen SER (emoción): "
            f"Accuracy={metrics['accuracy']:.3f}  "
            f"BalancedAcc={metrics['balanced_accuracy']:.3f}  "
            f"F1-macro={metrics['f1_macro']:.3f}  "
            f"N={metrics['total']}"
        )
        cm = metrics["confusion_matrix"]
        print("Matriz de confusión (filas=ref, columnas=pred) - orden:", ", ".join(SER_LABELS))
        for i, lbl in enumerate(SER_LABELS):
            row_vals = " ".join(f"{int(v):4d}" for v in cm[i])
            print(f"{lbl:9s} {row_vals}")
    else:
        print("No se evaluó el acierto de emoción (no se pudo inferir ref_emotion).")

    def _p(vals: List[float]) -> Tuple[float, float, float]:
        pct = _percentiles(vals)
        return pct["p50"], pct["p90"], pct["p99"]

    p_stt = _p(lat_stt)
    p_ser = _p(lat_ser)
    p_resp = _p(lat_resp)
    p_tts = _p(lat_tts)
    p_total = _p(lat_total)

    print(
        f"Latencias STT (s):       P50={p_stt[0]:.3f}   P90={p_stt[1]:.3f}   P99={p_stt[2]:.3f}\n"
        f"Latencias SER (s):       P50={p_ser[0]:.3f}   P90={p_ser[1]:.3f}   P99={p_ser[2]:.3f}\n"
        f"Latencias respuesta (s): P50={p_resp[0]:.3f}  P90={p_resp[1]:.3f}  P99={p_resp[2]:.3f}\n"
        f"Latencias TTS (s):       P50={p_tts[0]:.3f}   P90={p_tts[1]:.3f}   P99={p_tts[2]:.3f}\n"
        f"Latencias Totales (s):   P50={p_total[0]:.3f} P90={p_total[1]:.3f} P99={p_total[2]:.3f}"
    )
    print(f"CSV guardado en: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación de STT y pipeline completo.")
    parser.add_argument("--mode", choices=["stt", "pipeline"], default="stt")
    parser.add_argument("--input", required=True, help="CSV de entrada con audio_path y opcional ref_text/ref_emotion.")
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    parser.add_argument("--use-ollama", action="store_true", help="Parafrasear la respuesta con Ollama en modo pipeline.")
    parser.add_argument("--use-tts", action="store_true", help="Generar audio TTS en modo pipeline.")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)

    t0 = time.perf_counter()
    if args.mode == "stt":
        evaluate_stt(input_csv, output_csv)
    else:
        evaluate_full_pipeline(input_csv, output_csv, use_ollama=args.use_ollama, use_tts=args.use_tts)
    print(f"Tiempo total de evaluación: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
