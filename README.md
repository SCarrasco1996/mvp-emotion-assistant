# Asistente conversacional generativo multimodal para el apoyo a la regulación emocional (TFE UNIR)

Repositorio del prototipo (MVP) desarrollado para el Trabajo Fin de Máster en Inteligencia Artificial (UNIR).
El sistema implementa un pipeline local extremo a extremo:

**Audio (voz) → STT (transcripción) → SER (emoción) → respuesta (plantilla / LLM local) → (opcional) TTS**

Emociones objetivo (5 clases): **alegría, ira, tristeza, miedo y neutro**.

> ⚠️ **Aviso importante (uso responsable)**
> Este proyecto es un prototipo académico orientado a apoyo y experimentación. **No proporciona diagnóstico clínico, tratamiento ni consejo médico.**
> Si estás en una situación de riesgo o malestar intenso, busca ayuda profesional o servicios de emergencia de tu zona.

---

## Estado / configuración recomendada

Configuración “activa” con la que se reportan los resultados principales en la memoria:

- **STT:** `faster-whisper` con **Whisper `large-v3`** (GPU, `float16`, `beam_size=5`)
- **SER:** fine-tuning de **`microsoft/wavlm-large`** (experimento activo: `ser_exp5`)
- **LLM local:** **Ollama** con `qwen2.5:14b` (modo chat vía `/api/chat`)
- **TTS:** `pyttsx3` (opcional, activado por configuración)

---

## Qué incluye y qué NO incluye este repositorio

### Incluye
- Código fuente completo del MVP (`src/`)
- Ficheros de dependencias (`requirements.txt`)
- Scripts de preparación y evaluación
- Audios propios para evaluación: `data/eval_audio/`
- Resultados ligeros (`results/*.csv`, `results/*_summary.txt`, matrices/figuras si aplican)

### No incluye
- Entorno virtual: `.venv/`
- Audios del dataset MESD: `data/audio/`
- Pesos grandes del modelo SER (no se versionan en Git): `model.safetensors` (disponible en Releases, ver sección “Modelo SER (pesos)”).

---

## Estructura del proyecto

```
mvp-emotion-assistant/
  src/
    config.py
    stt_whisper.py
    emotion_ser.py
    prepare_mesd_metadata.py
    responses.py
    pipeline.py
    eval_pipeline.py
    ui_gradio.py

  data/
    metadata.csv
    eval.csv
    (no incluido) audio/
    eval_audio/ (audios propios para evaluación interna)

  model/
    config.json
    preprocessor_config.json
    (no incluido en Git) model.safetensors (adjunto en la Release v1.1-tfe)

  results/
    stt_eval.csv
    stt_eval_summary.txt
    pipeline_eval_*.csv
    pipeline_eval_*_summary.txt
    ser_exp*/ (artefactos ligeros por experimento)
```

---

## Requisitos

- **Python 3.12 (x64)** recomendado (también debería funcionar con 3.10+ si las dependencias lo permiten)
- GPU NVIDIA (opcional, pero recomendada para rendimiento)
- (Opcional) **Ollama** para generación con LLM local

---

## Instalación (Windows)

### 1) Crear y activar entorno virtual
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Instalar dependencias
```powershell
pip install -r requirements.txt
```

### 3) (Opcional / recomendado) Asegurar que PyTorch use CUDA
En algunos entornos, PyTorch puede quedar instalado solo para CPU. Si quieres forzar CUDA (cu128), puedes:

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verificación rápida:
```powershell
@"
import torch
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
"@ | python -
```

---

## Instalación (Linux / macOS) [orientativo]

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> En Linux, si deseas GPU, instala la versión de PyTorch con CUDA adecuada para tu sistema (según la documentación oficial de PyTorch).

---

## Dataset (MESD) y preparación

Este repositorio **no incluye** los audios por tamaño. Debes descargar **Mexican Emotional Speech Database (MESD)** desde su fuente (por ejemplo, Mendeley Data) y colocar los WAV aquí:

```
data/audio/
```

En este proyecto se mapean las etiquetas originales a 5 clases:
- Happiness → alegria
- Anger → ira
- Sadness → tristeza
- Fear → miedo
- Neutral → neutro
- Disgust → **se descarta** (para mantener 5 clases)

### Generar `metadata.csv`
```bash
python -m src.prepare_mesd_metadata
```

---

## Modelo SER (pesos)

La carpeta `model/` debe contener:
- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

Por tamaño, **`model.safetensors` no se incluye en el historial Git del repositorio**.  
Los pesos del modelo SER están disponibles como **archivo adjunto** en la sección **Releases** del repositorio.

**Descarga e instalación:**
1) Descarga `model.safetensors` desde la Release **`v1.1-tfe`**:  
   https://github.com/SCarrasco1996/mvp-emotion-assistant/releases/tag/v1.1-tfe
2) Copia el archivo descargado a la ruta:
   `model/`

---

## Configuración clave (`src/config.py`)

Parámetros típicos a ajustar:
- `use_ollama = True/False`
- `ollama_base_url = "http://localhost:11434"`
- `ollama_model = "qwen2.5:14b"`  (debe coincidir con `ollama list`)
- `settings.enable_tts = True/False`
- `settings.ser_encoder_name` (relevante si quieres reproducir experimentos históricos)
- Parámetros de Whisper (`large-v3`, `beam_size`, `compute_type`, device)

---

## LLM local con Ollama (opcional)

### 1) Instalar Ollama
- Windows: https://ollama.com/download/windows

### 2) Descargar el modelo
```bash
ollama pull qwen2.5:14b
```

### 3) Comprobar que está disponible
```bash
ollama list
```

### 4) Ajustar config
En `src/config.py`:
```python
use_ollama = True
ollama_base_url = "http://localhost:11434"
ollama_model = "qwen2.5:14b"
```

> En la UI no hay checkbox para activar/desactivar LLM: se controla desde `config.py`.

---

## Ejecución del MVP (interfaz Gradio)

```bash
python -m src.ui_gradio
```

La interfaz permite:
- Grabar o subir audio (normalización a mono; máximo ~20s según configuración)
- Ver transcripción STT
- Ver emoción estimada y probabilidades por clase
- Chat (con Ollama si está activo; fallback a plantillas si falla)
- Mostrar latencias por etapa (STT, SER, respuesta, TTS)

**Confianza baja:** se marca si `emotion_score < 0.35` (umbral en `ui_gradio.py`).

---

## Entrenamiento SER (opcional)

Entrenamiento genérico:
```bash
python -m src.emotion_ser --mode train [parámetros]
```

Notas de reproducibilidad:
- El encoder depende de `settings.ser_encoder_name` en `src/config.py`.
- Experimentos históricos:
  - exp1–exp3: `facebook/wav2vec2-xls-r-300m`
  - exp4–exp5: `microsoft/wavlm-large`

---

## Evaluación (`src/eval_pipeline.py`)

Ejemplos (Windows; ajusta rutas según tu sistema):

### 1) Evaluación STT
```powershell
python -m src.eval_pipeline --mode stt --input .\data\eval.csv --output .\results\stt_eval.csv
```

### 2) Pipeline completo (sin LLM)
```powershell
python -m src.eval_pipeline --mode pipeline --input .\data\eval.csv --output .\results\pipeline_eval_serexpX.csv
```

### 3) Pipeline completo (con LLM)
```powershell
python -m src.eval_pipeline --mode pipeline --input .\data\eval.csv --output .\results\pipeline_eval_ollama_serexpX.csv --use-ollama
```

### 4) Pipeline completo (con LLM + TTS)
```powershell
python -m src.eval_pipeline --mode pipeline --input .\data\eval.csv --output .\results\pipeline_eval_ollama_tts_serexpX.csv --use-ollama --use-tts
```

---

## Resultados reportados en la memoria (resumen)

Resultados principales (validación interna):
- **STT (conjunto interno, 20 audios):** WER medio **0,040**; CER medio **0,007**.
- **SER (ser_exp5, validación cruzada):** F1-macro medio **0,7413 ± 0,1204**.
- **Latencia E2E (percentiles, escenario con LLM):**
  - Con LLM, sin TTS: **P50 ≈ 4,890 s**
  - Con LLM + TTS: **P50 ≈ 5,077 s**
  - El mayor cuello de botella es la etapa **LLM**.

> Para detalles completos, consultar la memoria del TFE.

---

## Consideraciones de seguridad

- El modo de generación (Ollama) está diseñado para **parafrasear / generar respuestas breves** y se aplica un control de seguridad básico (longitud y filtros de términos sensibles).
- Si la generación falla o se considera no apropiada, el sistema devuelve una **plantilla** asociada a la emoción detectada.

---

## Licencia

Ver archivo `LICENSE`. La licencia aplica al **código** del repositorio; el dataset y modelos de terceros mantienen sus propias licencias.
