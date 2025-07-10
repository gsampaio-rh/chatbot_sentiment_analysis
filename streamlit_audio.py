import streamlit as st
import librosa
import numpy as np
import torch
import tempfile
import matplotlib.pyplot as plt
import spacy
import json
import subprocess
import time
import logging
from pathlib import Path
from transformers import pipeline

from faster_whisper import WhisperModel

# ===============================================================
# Modelos disponíveis no Hub → dict <apelido> : <repo HF>
# ===============================================================
_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# ===============================================================
# Utilidades
# ===============================================================


def convert_to_wav(input_path: str) -> str:
    """Converte qualquer áudio compatível com FFmpeg para mono WAV 16 kHz."""
    output_path = Path(input_path).with_suffix(".wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_path)


def format_timestamp(seconds: float) -> str:
    """Formata segundos para HH:MM:SS.mmm."""
    millis = int((seconds - int(seconds)) * 1000)
    return time.strftime("%H:%M:%S", time.gmtime(seconds)) + f".{millis:03d}"


# ===============================================================
# Configuração de LOGS → Streamlit
# ===============================================================
log_placeholder = st.empty()


class StreamlitLogHandler(logging.Handler):
    """Handler que envia logs para um widget Streamlit em tempo real."""

    def __init__(self, placeholder, max_lines: int = 400):
        super().__init__()
        self.placeholder = placeholder
        self.lines: list[str] = []
        self.max_lines = max_lines
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s ⟩ %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record):
        self.lines.append(self.format(record))
        self.lines = self.lines[-self.max_lines :]
        self.placeholder.code("\n".join(self.lines), language="text")


fw_logger = logging.getLogger("faster_whisper")
fw_logger.setLevel(logging.DEBUG)
fw_logger.addHandler(StreamlitLogHandler(log_placeholder))

# ===============================================================
# Barra lateral – Configurações
# ===============================================================

st.set_page_config(page_title="Voice Intelligence – faster-whisper", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurações")

    input_mode = st.radio(
        "Tipo de Entrada:",
        ("MP3/WAV", "Arquivo de Transcrição (.json)", "Exemplo Interno"),
    )

    sentiment_choice = st.radio(
        "Modelo de Sentimento:", ("spaCy local", "Transformers")
    )

    model_key = st.selectbox(
        "Modelo faster-whisper",
        list(_MODELS.keys()),
        index=list(_MODELS.keys()).index("large-v3"),
    )

    language = st.selectbox(
        "Idioma do áudio",
        options=[
            "pt",
            "en",
            "es",
            "fr",
            "de",
            "it",
            "ru",
            "zh",
            "ja",
            "ar",
            "hi",
            "ko",
        ],
        format_func=lambda x: {
            "pt": "Português",
            "en": "Inglês",
            "es": "Espanhol",
            "fr": "Francês",
            "de": "Alemão",
            "it": "Italiano",
            "ru": "Russo",
            "zh": "Chinês",
            "ja": "Japonês",
            "ar": "Árabe",
            "hi": "Hindi",
            "ko": "Coreano",
        }.get(x, x),
        index=0,
    )

    device_choice = st.selectbox(
        "Dispositivo de inferência",
        ["Auto", "cuda", "mps", "cpu"],
        index=0,
        format_func=lambda x: {
            "Auto": "Automático",
            "cuda": "CUDA (GPU)",
            "mps": "MPS (Apple Silicon)",
            "cpu": "CPU",
        }.get(x, x),
    )

    compute_choice = st.selectbox(
        "Precisão (compute_type)",
        ["Auto", "float16", "int8", "int8_float16"],
        index=0,
    )

    stream_live = st.checkbox("Mostrar transcrição em tempo real", value=True)

# ===============================================================
# Modelos de Sentimento & Intenção
# ===============================================================

nlp_sentiment = spacy.load("spacy_model/model-best")
intent_model = pipeline("text-classification", model="bert-base-uncased")

# ===============================================================
# Session State
# ===============================================================

for key, default in {
    "conversation": [],
    "sentiments": [],
    "intent": None,
    "audio_path": None,
    "device_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===============================================================
# Funções auxiliares de modelo
# ===============================================================


def resolve_device(device_choice: str) -> str:
    if device_choice == "Auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_choice


def resolve_compute_type(device: str, compute_choice: str) -> str:
    if compute_choice != "Auto":
        return compute_choice
    # Heurística padrão
    if device in {"cuda", "mps"}:
        return "float16"
    return "int8"


@st.cache_resource(show_spinner="🔄 Carregando modelo faster-whisper…")
def load_whisper(model_size: str, device_choice: str, compute_choice: str):
    device = resolve_device(device_choice)
    compute_type = resolve_compute_type(device, compute_choice)

    # Validação rápida
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA não disponível—trocando para CPU.")
        device = "cpu"
        compute_type = "int8"
    if device == "mps" and not torch.backends.mps.is_available():
        st.warning("MPS não disponível—trocando para CPU.")
        device = "cpu"
        compute_type = "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    model.meta_device = device
    model.meta_compute = compute_type
    return model

def transcribe_with_progress(
    audio_path: str,
    model_size: str,
    language: str,
    device_choice: str,
    compute_choice: str,
    stream_live: bool = True,
):
    model = load_whisper(model_size, device_choice, compute_choice)

    pbar = st.progress(0, text="Transcrevendo…")
    txt_placeholder = st.empty()

    segments, info = model.transcribe(
        audio=audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        log_progress=True,
        language=language,
    )

    processed_segments: list[dict] = []
    total = info.duration or 0.0

    for seg in segments:
        start_ts = format_timestamp(seg.start)
        end_ts = format_timestamp(seg.end)
        text = seg.text.strip()

        if stream_live:
            partial_text = ""
            for char in text:
                partial_text += char
                current_display = processed_segments + [
                    {"start": start_ts, "end": end_ts, "text": partial_text}
                ]
                txt_placeholder.code(
                    "\n".join(
                        [
                            f"[{s['start']} → {s['end']}] {s['text']}"
                            for s in current_display
                        ]
                    )
                )
                time.sleep(0.01)  # Ajuste o tempo se quiser mais rápido/lento
            # Após o streaming, registra o segmento completo (sem reexibir)
            processed_segments.append({"start": start_ts, "end": end_ts, "text": text})
        else:
            # Modo batch: apenas salva e atualiza o display no final
            processed_segments.append({"start": start_ts, "end": end_ts, "text": text})
            txt_placeholder.code(
                "\n".join(
                    [
                        f"[{s['start']} → {s['end']}] {s['text']}"
                        for s in processed_segments
                    ]
                )
            )

        if total:
            pbar.progress(min(seg.end / total, 1.0))

    pbar.empty()
    return processed_segments, model.meta_device, model.meta_compute


# ===============================================================
#  Interface principal
# ===============================================================

st.title("🎙️ Voice Intelligence App – faster-whisper")
st.markdown(
    "Upload de áudio ou texto para análise de transcrição, sentimento e intenção."
)

if input_mode == "MP3/WAV":
    uploaded_file = st.file_uploader(
        "🔊 Envie um arquivo de áudio", type=["wav", "mp3"]
    )
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="." + uploaded_file.name.split(".")[-1]
        ) as tmp_in:
            tmp_in.write(uploaded_file.read())
            raw_path = tmp_in.name
        wav_path = convert_to_wav(raw_path)
        st.session_state.audio_path = wav_path
        st.success("✅ Áudio carregado e convertido para WAV!")

        y, sr = librosa.load(wav_path, sr=None)
        plt.figure(figsize=(10, 1))
        plt.plot(y)
        plt.title("Forma de onda do áudio")
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude")
        st.pyplot(plt.gcf())

elif input_mode == "Arquivo de Transcrição (.json)":
    uploaded_json = st.file_uploader(
        "📄 Envie o arquivo de transcrição (.json)", type=["json"]
    )
    if uploaded_json:
        st.session_state.conversation = json.load(uploaded_json)

elif input_mode == "Exemplo Interno":
    st.session_state.conversation = [
        {
            "speaker": "Segment 1",
            "start": "00:00:00.000",
            "end": "00:00:03.000",
            "text": "Oi, estou com um problema na minha conta.",
        },
        {
            "speaker": "Segment 2",
            "start": "00:00:03.000",
            "end": "00:00:06.000",
            "text": "Claro, posso verificar isso para você.",
        },
        {
            "speaker": "Segment 3",
            "start": "00:00:06.000",
            "end": "00:00:08.000",
            "text": "Quero cancelar o serviço.",
        },
    ]

# ------------------- CONTROLES ----------------------
if input_mode == "MP3/WAV" and st.session_state.audio_path:
    # ========== TOP ROW: 3 buttons ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        transcribe_clicked = st.button("🎙 Transcrever Áudio")

    with col2:
        sentiment_clicked = st.button("🧠 Analisar Sentimento")

    with col3:
        intent_clicked = st.button("🎯 Detectar Intenção")

    if transcribe_clicked:
        start = time.time()
        with st.spinner("Transcrevendo…"):
            segments, device_name, compute_type = transcribe_with_progress(
                st.session_state.audio_path,
                model_key,
                language,
                device_choice,
                compute_choice,
                stream_live,
            )
            st.session_state.device_name = device_name
            st.session_state.compute_type = compute_type
            st.session_state.conversation = [
                {
                    "speaker": f"Segment {i+1}",
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for i, seg in enumerate(segments)
            ]
        st.success(f"✅ Transcrição concluída em {time.time() - start:.2f}s")
        st.markdown(f"**Dispositivo:** `{st.session_state.device_name}`")

    # ---------- SENTIMENTO ----------
    if sentiment_clicked:
        start = time.time()
        results = []
        with st.spinner("Analisando sentimentos…"):
            for turn in st.session_state.conversation:
                if sentiment_choice == "spaCy local":
                    doc = nlp_sentiment(turn["text"])
                    cats = doc.cats
                    label = max(cats, key=cats.get)
                    score = cats[label]
                else:
                    sent_model = pipeline(
                        "text-classification",
                        model="pysentimiento/bertweet-pt-sentiment",
                    )
                    r = sent_model(turn["text"])[0]
                    label = r["label"]
                    score = r["score"]
                results.append(
                    {
                        "speaker": turn["speaker"],
                        "text": turn["text"],
                        "label": label,
                        "score": score,
                    }
                )
        st.session_state.sentiments = results
        st.success(f"✅ Sentimentos analisados em {time.time() - start:.2f}s")

    # ---------- INTENÇÃO ----------
    if intent_clicked:
        start = time.time()
        full_text = " ".join([x["text"] for x in st.session_state.conversation])
        with st.spinner("Detectando intenção…"):
            result = intent_model(full_text)[0]
        st.session_state.intent = result
        st.success(f"✅ Intenção detectada em {time.time() - start:.2f}s")

# ------------------- EXIBIÇÃO DA CONVERSA ----------------------
if st.session_state.conversation:
    st.subheader("💬 Conversa")

    bubble_color = "#f0f0f0"

    for turn in st.session_state.conversation:
        sentiment_info = next(
            (
                f"<br><small>Sentimento: {s['label']} ({s['score']*100:.1f}%)</small>"
                for s in st.session_state.sentiments
                if s["text"] == turn["text"]
            ),
            "",
        )

        timestamp_info = f"<small>{turn['start']} → {turn['end']}</small><br>"

        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; padding: 4px 0;'>
                <div style='background-color: {bubble_color}; padding: 10px 16px; border-radius: 12px; max-width: 90%;'>
                    <strong>{timestamp_info}</strong>{turn['text']}{sentiment_info}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.intent:
        st.subheader("🎯 Intenção detectada")
        st.markdown(
            f"**{st.session_state.intent['label']}** ({st.session_state.intent['score']*100:.1f}%)"
        )

# ------------------- ESTILO GLOBAL ----------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #f5f5f5; color: #2c2c2c; font-family: 'Helvetica Neue', sans-serif; }
        .css-1aumxhk, .stContainer { background-color: #ffffff; border-radius: 16px; padding: 16px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); }
        .block-container { padding: 2rem 4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
