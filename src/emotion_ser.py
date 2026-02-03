"""Clasificador de emociones (SER): inferencia y entrenamiento.

Incluye:
- Carga del modelo ajustado (fine-tuned) y predicción de emoción sobre audio.
- Entrenamiento con partición K-Fold y métricas (F1-macro, matriz de confusión).

El módulo se utiliza tanto desde el pipeline del MVP como desde la línea de
comandos para ejecutar experimentos reproducibles.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

from .config import settings


@dataclass
class EmotionPrediction:
    label: str
    score: float
    scores_per_class: Dict[str, float]
    latency: float


class EmotionSER:
    """
    Clasificador de emociones en voz (5 clases) usando un modelo
    de audio preentrenado + ajuste fino (guardado en la carpeta ./model).
    """

    @staticmethod
    def _normalize_id2label(mapping: dict | list | None) -> Dict[int, str]:
        """Convierte id2label a un diccionario con claves int para evitar KeyError."""
        if mapping is None:
            return {}
        if isinstance(mapping, dict):
            return {int(k): str(v) for k, v in mapping.items()}
        if isinstance(mapping, list):
            return {i: str(lbl) for i, lbl in enumerate(mapping)}
        return {}

    def __init__(self) -> None:
        self.device = torch.device(settings.device)

        model_path = settings.ser_model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se ha encontrado el modelo SER en {model_path}.\n"
                "Primero debe entrenarse."
            )

        if settings.debug:
            print(
                f"[EmotionSER] Cargando modelo SER desde {model_path} "
                f"en dispositivo {self.device}"
            )

        self.config = AutoConfig.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.id2label = self._normalize_id2label(self.config.id2label)
        self.label2id = {str(v): int(k) for k, v in self.id2label.items()}

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        audio, _ = librosa.load(
            str(audio_path),
            sr=settings.ser_sampling_rate,
            mono=True,
        )
        return audio

    @torch.inference_mode()
    def predict(self, audio_path: str | Path) -> EmotionPrediction:
        # Se realiza la inferencia sin gradientes para minimizar latencia y consumo de memoria.
        start_time = time.perf_counter()

        audio = self._load_audio(audio_path)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=settings.ser_sampling_rate,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        latency = time.perf_counter() - start_time

        best_idx = int(np.argmax(probs))
        best_label = self.id2label.get(best_idx, str(best_idx))
        best_score = float(probs[best_idx])

        scores_per_class = {self.id2label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

        return EmotionPrediction(
            label=best_label,
            score=best_score,
            scores_per_class=scores_per_class,
            latency=latency,
        )


# -----------------------------------------------------------------------
# Entrenamiento del modelo SER sobre MESD con validación cruzada (K-Fold)
# -----------------------------------------------------------------------


def _build_metadata_example():
    """
    Carga el CSV de MESD y construye un DataFrame con:
      - audio_path: ruta absoluta al fichero .wav
      - label: una de las 5 etiquetas (alegria, ira, tristeza, miedo, neutro)
    """
    import pandas as pd

    base_data = settings.data_dir
    csv_path = base_data / "metadata.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {csv_path}."
        )

    df_mesd = pd.read_csv(csv_path)

    map_mesd = {
        "Happiness": "alegria",
        "Anger": "ira",
        "Sadness": "tristeza",
        "Fear": "miedo",
        "Neutral": "neutro",
    }

    df_mesd = df_mesd[df_mesd["emotion_raw"].isin(map_mesd.keys())].copy()
    df_mesd["label"] = df_mesd["emotion_raw"].map(map_mesd)

    df_mesd["audio_path"] = df_mesd["file"].apply(
        lambda x: str(base_data / "audio" / x)
    )

    df_mesd = df_mesd[["audio_path", "label"]].reset_index(drop=True)
    return df_mesd


def train_ser_model(
    output_dir: Path | None = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    num_folds: int = 5,
    weight_decay: float = 0.01,
    augment_train: bool = False,
    early_stopping_patience: int | None = None,
):
    """
    Entrena un modelo SER sobre MESD mediante validación cruzada estratificada y, opcionalmente, con aumento de datos controlado.
    """
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch import optim

    # Se fijan semillas para obtener resultados reproducibles.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if output_dir is None:
        output_dir = settings.ser_model_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _build_metadata_example()
    print(f"Total de muestras MESD (después del filtrado): {len(df)}")
    print("Distribución por etiqueta:")
    print(df["label"].value_counts())

    labels_np = df["label"].values

    label2id = settings.ser_label2id
    id2label = {v: k for k, v in label2id.items()}

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        settings.ser_encoder_name
    )
    base_config = AutoConfig.from_pretrained(settings.ser_encoder_name)
    base_config.num_labels = len(label2id)
    base_config.label2id = label2id
    base_config.id2label = id2label

    def augment_audio(audio: np.ndarray, sr: int) -> np.ndarray:
        """Aplica transformaciones simples para aumentar la variabilidad."""
        if audio.size == 0:
            return audio

        y = audio.astype(np.float32)

        if np.random.rand() < 0.5:
            rate = float(np.random.uniform(0.9, 1.1))
            try:
                y = librosa.effects.time_stretch(y, rate=rate)
            except Exception:
                pass

        if np.random.rand() < 0.5:
            n_steps = float(np.random.uniform(-1.0, 1.0))
            try:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            except Exception:
                pass

        if np.random.rand() < 0.5:
            noise_amp = 0.005 * (np.abs(y).max() + 1e-9) * np.random.uniform(0.5, 1.5)
            noise = noise_amp * np.random.normal(size=y.shape[0])
            y = y + noise

        max_val = np.max(np.abs(y)) + 1e-9
        if max_val > 1.0:
            y = y / max_val

        return y.astype(np.float32)

    class EmotionDataset(Dataset):
        def __init__(self, df: pd.DataFrame, augment: bool = False):
            self.df = df.reset_index(drop=True)
            self.augment = augment

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row = self.df.iloc[idx]
            audio_path = row["audio_path"]
            label = row["label"]

            audio, _ = librosa.load(
                audio_path, sr=settings.ser_sampling_rate, mono=True
            )

            if self.augment:
                audio = augment_audio(audio, settings.ser_sampling_rate)

            features = feature_extractor(
                audio,
                sampling_rate=settings.ser_sampling_rate,
                return_tensors="pt",
            )

            item = {
                "input_values": features["input_values"].squeeze(0),
                "labels": label2id[label],
            }
            if "attention_mask" in features:
                item["attention_mask"] = features["attention_mask"].squeeze(0)
            return item

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [b["input_values"] for b in batch]
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)

        padded = feature_extractor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
        )

        attention_mask = padded.get("attention_mask", None)

        result = {
            "input_values": padded["input_values"],
            "labels": labels,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return {k: v.to(settings.device) for k, v in result.items()}

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_best_f1: List[float] = []
    fold_best_balanced_acc: List[float] = []
    fold_best_top2: List[float] = []
    best_global_f1 = 0.0

    # Se definen acumuladores globales para informes agregados (matriz de confusión y reportes).
    global_labels: List[int] = []
    global_preds: List[int] = []

    for fold, (train_index, val_index) in enumerate(
        skf.split(np.zeros(len(labels_np)), labels_np), start=1
    ):
        print(f"\n========== Fold {fold}/{num_folds} ==========")
        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)

        model = AutoModelForAudioClassification.from_pretrained(
            settings.ser_encoder_name,
            config=base_config,
        ).to(settings.device)

        train_dataset = EmotionDataset(train_df, augment=augment_train)
        val_dataset = EmotionDataset(val_df, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_f1_fold = 0.0
        best_balanced_fold = 0.0
        best_top2_fold = 0.0
        best_preds_fold: List[int] = []
        best_labels_fold: List[int] = []
        best_epoch = 0
        epochs_no_improve = 0
        use_early_stopping = (
            early_stopping_patience is not None and early_stopping_patience > 0
        )

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_values=batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / max(1, len(train_loader))

            model.eval()
            all_preds: List[int] = []
            all_labels: List[int] = []
            top2_correct_epoch = 0

            with torch.inference_mode():
                for batch in val_loader:
                    outputs = model(
                        input_values=batch["input_values"],
                        attention_mask=batch.get("attention_mask"),
                    )
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    labels_batch = batch["labels"].cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(labels_batch)
                    # Se calcula la exactitud top-2 en la misma pasada de inferencia.
                    topk = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).indices.cpu().numpy()
                    for i, lbl in enumerate(labels_batch):
                        if lbl in topk[i]:
                            top2_correct_epoch += 1

            f1_macro = f1_score(all_labels, all_preds, average="macro")
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            top2_acc = top2_correct_epoch / max(1, len(all_labels))

            print(
                f"Fold {fold} - Epoch {epoch+1}/{num_epochs} "
                f"- loss: {avg_train_loss:.4f} - F1-macro: {f1_macro:.4f} "
                f"- BalancedAcc: {balanced_acc:.4f} - Top2: {top2_acc:.4f}"
            )

            if f1_macro > best_f1_fold:
                best_f1_fold = f1_macro
                best_balanced_fold = balanced_acc
                best_top2_fold = top2_acc
                best_preds_fold = list(all_preds)
                best_labels_fold = list(all_labels)
                best_epoch = epoch + 1
                epochs_no_improve = 0
            elif use_early_stopping:
                epochs_no_improve += 1

            if f1_macro > best_global_f1:
                best_global_f1 = f1_macro
                print(
                    f"  >>> Nuevo mejor F1-macro global: {best_global_f1:.4f}. "
                    f"Guardando modelo en {output_dir}"
                )
                model.save_pretrained(output_dir)
                feature_extractor.save_pretrained(output_dir)
                base_config.save_pretrained(output_dir)

            if use_early_stopping and epochs_no_improve >= early_stopping_patience:
                print(
                    f"  >>> Early stopping en epoch {epoch+1} "
                    f"(mejor F1-macro fold: {best_f1_fold:.4f} en epoch {best_epoch})"
                )
                break

        fold_best_f1.append(best_f1_fold)
        fold_best_balanced_acc.append(best_balanced_fold)
        fold_best_top2.append(best_top2_fold)
        global_labels.extend(best_labels_fold)
        global_preds.extend(best_preds_fold)
        print(f"Mejor F1-macro del fold {fold}: {best_f1_fold:.4f}")
        print(f"Mejor BalancedAcc del fold {fold}: {best_balanced_fold:.4f}")
        print(f"Mejor Top-2 acc del fold {fold}: {best_top2_fold:.4f}")

    mean_f1 = float(np.mean(fold_best_f1))
    std_f1 = float(np.std(fold_best_f1))
    print("\n===== Resumen K-Fold =====")
    print("F1-macro por fold:", [f"{x:.4f}" for x in fold_best_f1])
    print(f"F1-macro medio: {mean_f1:.4f} ± {std_f1:.4f}")
    print("BalancedAcc por fold:", [f"{x:.4f}" for x in fold_best_balanced_acc])
    print("Top-2 acc por fold:", [f"{x:.4f}" for x in fold_best_top2])
    print(
        f"Mejor F1-macro global (modelo guardado en {output_dir}): {best_global_f1:.4f}"
    )

    # ------------------------
    # Reportes agregados finales
    # ------------------------
    if len(global_labels) == 0:
        return

    class_names = [id2label[i] for i in range(len(id2label))]

    cm = confusion_matrix(global_labels, global_preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)

    report_dict = classification_report(
        global_labels,
        global_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_path, index=True)

    cm_raw_path = output_dir / "confusion_matrix_raw.csv"
    cm_norm_path = output_dir / "confusion_matrix_norm.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_raw_path)
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(cm_norm_path)

    # Se estima el IC del 95% de F1-macro mediante bootstrap.
    def bootstrap_ci(
        labels: List[int],
        preds: List[int],
        num_samples: int = 1000,
        alpha: float = 0.95,
    ) -> Dict[str, float]:
        metrics: List[float] = []
        labels_arr = np.array(labels)
        preds_arr = np.array(preds)
        for _ in range(num_samples):
            idx = np.random.randint(0, len(labels_arr), len(labels_arr))
            metrics.append(
                f1_score(labels_arr[idx], preds_arr[idx], average="macro")
            )
        lower = np.percentile(metrics, (1 - alpha) / 2 * 100)
        upper = np.percentile(metrics, (1 + alpha) / 2 * 100)
        return {"mean": float(np.mean(metrics)), "low": float(lower), "high": float(upper)}

    ci_f1 = bootstrap_ci(global_labels, global_preds)
    print(
        f"IC 95% bootstrap F1-macro (global): "
        f"{ci_f1['mean']:.4f} [{ci_f1['low']:.4f}, {ci_f1['high']:.4f}]"
    )

    # Se guarda la figura de la matriz de confusión normalizada.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Etiqueta real",
        xlabel="Predicción",
        title="Matriz de confusión (normalizada)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    cm_fig_path = output_dir / "confusion_matrix_norm.png"
    plt.savefig(cm_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Reportes guardados en:\n - {report_path}\n - {cm_raw_path}\n - {cm_norm_path}\n - {cm_fig_path}")


def evaluate_ser_pretrained_baseline(
    num_folds: int = 5,
    batch_size: int = 8,
) -> None:
    """
    Evalúa el encoder preentrenado (sin ajuste fino).
    """
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df = _build_metadata_example()
    print(f"Total de muestras MESD (después del filtrado): {len(df)}")
    labels_np = df["label"].values

    label2id = settings.ser_label2id
    id2label = {v: k for k, v in label2id.items()}

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        settings.ser_encoder_name
    )
    base_config = AutoConfig.from_pretrained(settings.ser_encoder_name)
    base_config.num_labels = len(label2id)
    base_config.label2id = label2id
    base_config.id2label = id2label

    class EmotionDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row = self.df.iloc[idx]
            audio_path = row["audio_path"]
            label = row["label"]

            audio, _ = librosa.load(
                audio_path, sr=settings.ser_sampling_rate, mono=True
            )

            features = feature_extractor(
                audio,
                sampling_rate=settings.ser_sampling_rate,
                return_tensors="pt",
            )

            item = {
                "input_values": features["input_values"].squeeze(0),
                "labels": label2id[label],
            }
            if "attention_mask" in features:
                item["attention_mask"] = features["attention_mask"].squeeze(0)
            return item

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [b["input_values"] for b in batch]
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)

        padded = feature_extractor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
        )

        attention_mask = padded.get("attention_mask", None)

        result = {
            "input_values": padded["input_values"],
            "labels": labels,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return {k: v.to(settings.device) for k, v in result.items()}

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_f1: List[float] = []

    print("\nEvaluando modelo preentrenado sin ajuste fino (línea base)...")
    for fold, (_, val_index) in enumerate(
        skf.split(np.zeros(len(labels_np)), labels_np), start=1
    ):
        val_df = df.iloc[val_index].reset_index(drop=True)
        val_dataset = EmotionDataset(val_df)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = AutoModelForAudioClassification.from_pretrained(
            settings.ser_encoder_name,
            config=base_config,
        ).to(settings.device)
        model.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.inference_mode():
            for batch in val_loader:
                outputs = model(
                    input_values=batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_batch = batch["labels"].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels_batch)

        f1_macro = f1_score(all_labels, all_preds, average="macro")
        fold_f1.append(f1_macro)
        print(f"Fold {fold} - F1-macro (sin ajuste fino): {f1_macro:.4f}")

    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1))
    print("\n===== Resumen de línea base (sin ajuste fino) =====")
    print("F1-macro por fold:", [f"{x:.4f}" for x in fold_f1])
    print(f"F1-macro medio: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena o evalúa el modelo SER."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "baseline"],
        default="train",
        help="train: ajuste fino con MESD; baseline: evalúa sin ajuste fino.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directorio donde se guardará el modelo y los reportes. "
            "Por defecto se usa settings.ser_model_path."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento (por defecto: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño de lote para entrenamiento/evaluación (por defecto: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Tasa de aprendizaje de AdamW (por defecto: 3e-5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Decaimiento de pesos (weight decay) de AdamW (por defecto: 0.01).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Número de folds para la validación cruzada (por defecto: 5).",
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Activa el aumento de datos en el conjunto de entrenamiento.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Número de épocas sin mejora tras las cuales se aplica la parada temprana (early stopping)."
        ),
    )

    args = parser.parse_args()

    try:
        if args.mode == "train":
            print("Lanzando entrenamiento SER (MESD, K-Fold)...")
            train_ser_model(
                output_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_folds=args.num_folds,
                weight_decay=args.weight_decay,
                augment_train=args.augment_train,
                early_stopping_patience=args.early_stopping_patience,
            )
            print("Entrenamiento finalizado.")
        else:
            print("Lanzando evaluación de línea base (modelo sin ajuste fino)...")
            evaluate_ser_pretrained_baseline(
                num_folds=args.num_folds,
                batch_size=args.batch_size,
            )
            print("Evaluación de línea base finalizada.")
    except Exception as exc:
        print(f"Error durante el proceso: {exc}")
        raise
