"""Generación de `metadata.csv` para MESD.

A partir de los ficheros WAV ubicados en `data/audio/`, se construye un CSV con
metadatos mínimos extraídos del nombre del archivo (etiqueta original, tipo de
voz, subcorpus y palabra). Este CSV se utiliza posteriormente en el pipeline de
preparación y entrenamiento.
"""

from __future__ import annotations

import csv

from .config import settings


def main() -> None:
    data_dir = settings.data_dir
    audio_dir = data_dir / "audio"
    output_csv = data_dir / "metadata.csv"

    rows: list[dict[str, str]] = []

    for wav_path in audio_dir.rglob("*.wav"):
        name = wav_path.stem  # Nombre sin extensión (p. ej., Anger_C_A_abajo).
        parts = name.split("_")

        # Formato esperado: <emotion>_<type>_<corpus>_<word...>.
        if len(parts) < 4:
            print(f"Ignorando nombre atípico (no se ajusta al patrón esperado): {name}")
            continue

        emotion = parts[0]  # Anger, Disgust, Fear, Happiness, Neutral, Sadness.
        voice_type = parts[1]  # F, M, C.
        corpus = parts[2]  # A, B.
        word = "_".join(parts[3:])  # Resto de la cadena (p. ej., 'por_favor').

        rows.append(
            {
                "file": wav_path.name,
                "emotion_raw": emotion,
                "voice_type": voice_type,
                "corpus": corpus,
                "word": word,
            }
        )

    fieldnames = ["file", "emotion_raw", "voice_type", "corpus", "word"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Guardadas {len(rows)} filas en {output_csv}")


if __name__ == "__main__":
    main()
