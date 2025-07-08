import streamlit as st
import librosa
import numpy as np
import torch
import tempfile
import matplotlib.pyplot as plt
import spacy
import json
import wave
import contextlib
from transformers import pipeline
import subprocess
import time
import logging
from pathlib import Path

from faster_whisper import WhisperModel  # ➜ nova lib

# ===============================================================
#  Modelos disponíveis no Hub → dict <apelido> : <repo HF>
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
#  Utilidades
# ===============================================================


def convert_to_wav(input_path: str) -> str:
    """Converte qualquer áudio compatível com FFmpeg para mono WAV 16 kHz."""
    output_path = Path(input_path).with_suffix(".wav")
    command = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz
        str(output_path),
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_path)


# ===============================================================
#  Configuração de LOGS → Streamlit
# ===============================================================
log_placeholder = st.empty()  # espaço para mostrar logs "verbose"


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


# Aponta o logger interno do faster-whisper para o handler acima
fw_logger = logging.getLogger("faster_whisper")
fw_logger.setLevel(logging.DEBUG)
fw_logger.addHandler(StreamlitLogHandler(log_placeholder))


# ===============================================================
#  Barra lateral – Configurações
# ===============================================================

st.set_page_config(page_title="Voice Intelligence – faster-whisper", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurações")

    input_mode = st.radio(
        "Tipo de Entrada:",
        ("MP3/WAV", "Arquivo de Transcrição (.json)", "Exemplo Interno"),
    )

    sentiment_choice = st.radio(
        "Modelo de Sentimento:", ("spaCy local", "Transformers (online)")
    )

    # Select com os apelidos dos modelos disponíveis
    model_key = st.selectbox(
        "Modelo faster-whisper",
        list(_MODELS.keys()),
        index=list(_MODELS.keys()).index("base"),
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

# ===============================================================
#  Modelos de Sentimento & Intenção
# ===============================================================

nlp_sentiment = spacy.load("spacy_model/model-best")
intent_model = pipeline("text-classification", model="bert-base-uncased")

# ===============================================================
#  Session State (inicializa chaves)
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
#  Funções de transcrição com barra de progresso
# ===============================================================


@st.cache_resource(show_spinner="🔄 Carregando modelo faster-whisper…")
def load_whisper(model_size: str):
    """Carrega o modelo apenas 1× (cache). Também anota em qual dispositivo."""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # Armazenamos o nome do dispositivo para uso posterior
    model.meta_device = device  # atributo extra, não existe na lib
    return model


def transcribe_with_progress(audio_path: str, model_size: str, language: str):
    model = load_whisper(model_size)

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

    transcript_lines = []
    total = None

    for seg in segments:
        transcript_lines.append(f"[{seg.start:7.2f}s → {seg.end:7.2f}s] {seg.text}")
        txt_placeholder.text("\n".join(transcript_lines))

        total = info.duration or total
        if total:
            pbar.progress(min(seg.end / total, 1.0))

    pbar.empty()
    return "\n".join(transcript_lines), model.meta_device


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
        {"speaker": "Speaker 1", "text": "Oi, estou com um problema na minha conta."},
        {"speaker": "Speaker 2", "text": "Claro, posso verificar isso para você."},
        {"speaker": "Speaker 1", "text": "Quero cancelar o serviço."},
    ]

# ------------------- CONTROLES ----------------------
if input_mode == "MP3/WAV" and st.session_state.audio_path:
    col1, col2, col3 = st.columns(3)

    # ---------- TRANSCRIÇÃO ----------
    with col1:
        if st.button("🎙 Transcrever Áudio"):
            start = time.time()
            with st.spinner("Transcrevendo…"):
                text, device_name = transcribe_with_progress(
                    st.session_state.audio_path, model_key, language
                )
                st.session_state.device_name = device_name
                st.session_state.conversation = [{"speaker": "Speaker 1", "text": text}]
            st.success(f"✅ Transcrição concluída em {time.time() - start:.2f}s")
            st.markdown(f"**Dispositivo:** `{st.session_state.device_name}`")

    # ---------- SENTIMENTO ----------
    with col2:
        if st.button(
            "🧠 Analisar Sentimento", disabled=not st.session_state.conversation
        ):
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
    with col3:
        if st.button(
            "🎯 Detectar Intenção", disabled=not st.session_state.conversation
        ):
            start = time.time()
            full_text = " ".join([x["text"] for x in st.session_state.conversation])
            with st.spinner("Detectando intenção…"):
                result = intent_model(full_text)[0]
            st.session_state.intent = result
            st.success(f"✅ Intenção detectada em {time.time() - start:.2f}s")

# ------------------- EXIBIÇÃO DA CONVERSA ----------------------
if st.session_state.conversation:
    st.subheader("💬 Conversa")

    speaker_colors = {
        "Speaker 1": "#e1ffc7",
        "Speaker 2": "#d2e3fc",
        "Speaker 3": "#ffe0e0",
        "Speaker 4": "#f0e68c",
        "Speaker 5": "#d1c4e9",
    }

    for turn in st.session_state.conversation:
        speaker = turn["speaker"]
        bubble_color = speaker_colors.get(speaker, "#eeeeee")
        alignment = "flex-start" if speaker.endswith("1") else "flex-end"

        sentiment_info = next(
            (
                f"<br><small>Sentimento: {s['label']} ({s['score']*100:.1f}%)</small>"
                for s in st.session_state.sentiments
                if s["text"] == turn["text"]
            ),
            "",
        )

        st.markdown(
            f"""
            <div style='display: flex; justify-content: {alignment}; padding: 4px 0;'>
                <div style='background-color: {bubble_color}; padding: 10px 16px; border-radius: 12px; max-width: 70%;'>
                    <strong>{speaker}</strong><br>{turn['text']}{sentiment_info}
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
