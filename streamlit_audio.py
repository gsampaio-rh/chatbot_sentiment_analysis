import streamlit as st
import librosa
import numpy as np
import torch
import whisper
import tempfile
import matplotlib.pyplot as plt
import spacy
import json
import datetime
import wave
import contextlib
from transformers import pipeline
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import subprocess
import os


def convert_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
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
        output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def whisper_diarize(audio_path, num_speakers=2, model_size="base", language="pt"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    st.write(f"Usando dispositivo: `{device}`")

    whisper_model = whisper.load_model(model_size)
    result = whisper_model.transcribe(audio_path, language=language)
    segments = result["segments"]

    with contextlib.closing(wave.open(audio_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device=device
    )
    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        if end - start < 0.5:
            return np.zeros((192,))
        try:
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_path, clip)
            if waveform.shape[0] > 1:
                waveform = waveform[0].unsqueeze(0)
            embedding = embedding_model(waveform[None].to(device))
            return embedding.squeeze()
        except Exception as e:
            st.warning(f"Falha ao extrair embedding do segmento ({start}-{end}s): {e}")
            return np.zeros((192,))

    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    if np.allclose(embeddings, embeddings[0]):
        st.warning(
            "Todos os embeddings são semelhantes. Diarização pode estar falhando."
        )
        for i in range(len(segments)):
            segments[i]["speaker"] = "Speaker 1"
        return segments

    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = f"Speaker {labels[i] + 1}"

    return segments


# ------------------- SIDEBAR ----------------------
st.set_page_config(page_title="Voice Intelligence App", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurações")
    input_mode = st.radio(
        "Tipo de Entrada:",
        ("MP3/WAV", "Arquivo de Transcrição (.json)", "Exemplo Interno"),
    )
    sentiment_choice = st.radio(
        "Modelo de Sentimento:", ("spaCy local", "Transformers (online)")
    )
    diariazacao = st.checkbox("Realizar diarização de falantes", value=True)
    num_speakers = st.slider("Número de falantes", 2, 5, 2)
    model_size = st.selectbox(
        "Tamanho do modelo Whisper",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
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

# ------------------- MODELS -----------------------
nlp_sentiment = spacy.load("spacy_model/model-best")
intent_model = pipeline("text-classification", model="bert-base-uncased")

# ------------------- SESSION STATE ----------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []
if "intent" not in st.session_state:
    st.session_state.intent = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

# ------------------- INTERFACE ----------------------
st.title("🎙️ Voice Intelligence App")
st.markdown(
    "Upload de conversa em áudio ou texto para análise manual de transcrição, sentimento e intenção."
)

if input_mode == "MP3/WAV":
    uploaded_file = st.file_uploader(
        "🔊 Envie um arquivo de áudio", type=["wav", "mp3"]
    )
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_path = tmp_file.name
        tmp_path = convert_to_wav(input_path)
        st.session_state.audio_path = tmp_path
        st.success("Audio uploaded and converted!")
        y, sr = librosa.load(tmp_path, sr=None)
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
        {"speaker": "Cliente", "text": "Oi, estou com um problema na minha conta."},
        {"speaker": "Atendente", "text": "Claro, posso verificar isso para você."},
        {"speaker": "Cliente", "text": "Quero cancelar o serviço."},
    ]

# ------------------- CONTROLES ----------------------
if input_mode == "MP3/WAV" and st.session_state.audio_path:
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎙 Transcrever Áudio"):
            with st.spinner("Transcrevendo com Whisper..."):
                if diariazacao:
                    diarized_segments = whisper_diarize(
                        st.session_state.audio_path, num_speakers, model_size, language
                    )
                    st.session_state.conversation = [
                        {"speaker": s["speaker"], "text": s["text"].strip()}
                        for s in diarized_segments
                    ]
                else:
                    model = whisper.load_model(model_size)
                    result = model.transcribe(
                        st.session_state.audio_path, language=language
                    )
                    st.session_state.conversation = [
                        {"speaker": "Speaker 1", "text": result["text"].strip()}
                    ]

    with col2:
        if st.button(
            "🧠 Analisar Sentimento", disabled=not st.session_state.conversation
        ):
            results = []
            with st.spinner("Analisando sentimentos..."):
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

    with col3:
        if st.button(
            "🎯 Detectar Intenção", disabled=not st.session_state.conversation
        ):
            full_text = " ".join([x["text"] for x in st.session_state.conversation])
            result = intent_model(full_text)[0]
            st.session_state.intent = result

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

        sentiment_info = ""
        for s in st.session_state.sentiments:
            if s["text"] == turn["text"]:
                sentiment_info = f"<br><small>Sentimento: {s['label']} ({s['score']*100:.1f}%)</small>"

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

# ------------------- ESTILO ----------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5f5;
            color: #2c2c2c;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .css-1aumxhk, .stContainer {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .block-container {
            padding: 2rem 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
