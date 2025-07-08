import streamlit as st
import librosa
import numpy as np
import torch
import torchaudio
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

# ------------------- FUNCTIONS --------------------
def format_time(secs):
    return str(datetime.timedelta(seconds=int(secs)))

def whisper_diarize(audio_path, num_speakers=2, model_size="base", language="pt"):

    # --- Device selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    st.write(f"Usando dispositivo: `{device}`")

    # --- Transcribe with Whisper ---
    st.info("Transcrevendo com Whisper...")
    whisper_model = whisper.load_model(model_size)
    result = whisper_model.transcribe(audio_path, language=language)
    segments = result["segments"]

    # --- Get duration safely using torchaudio ---
    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
    except Exception as e:
        st.error(f"Erro ao calcular duração do áudio: {e}")
        return []

    # --- Embedding model ---
    st.info("Extraindo embeddings de falantes...")
    try:
        embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb", device=device
        )
        audio = Audio()
    except Exception as e:
        st.error(f"Erro ao carregar modelo de embedding: {e}")
        return []

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

    # --- Get embeddings ---
    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    # --- Verifica uniformidade dos embeddings ---
    if np.allclose(embeddings, embeddings[0]):
        st.warning(
            "Todos os embeddings são semelhantes. Diarização pode estar falhando."
        )
        for i in range(len(segments)):
            segments[i]["speaker"] = "Speaker 1"
        return segments

    # --- Clustering ---
    try:
        clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = f"Speaker {labels[i] + 1}"
    except Exception as e:
        st.error(f"Erro na clusterização dos embeddings: {e}")
        for i in range(len(segments)):
            segments[i]["speaker"] = "Speaker 1"

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

# ------------------- INTERFACE PRINCIPAL ----------------------
st.title("🎙️ Voice Intelligence App")
st.markdown(
    "Upload de conversa em áudio ou texto para análise automática de sentimento e intenção com experiência visual refinada."
)

conversation = []

if input_mode == "MP3/WAV":
    uploaded_file = st.file_uploader(
        "🔊 Envie um arquivo de áudio", type=["wav", "mp3"]
    )
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            # Convert to WAV if needed
            audio_path = tmp.name
            input_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())
            if not input_path.endswith(".wav"):
                subprocess.call(["ffmpeg", "-i", input_path, audio_path, "-y"])
            else:
                audio_path = input_path

        st.success("Audio uploaded and converted!")

        with st.spinner("🔈 Carregando e processando áudio..."):
            y, sr = librosa.load(audio_path, sr=None)
            plt.figure(figsize=(10, 1))
            plt.plot(y)
            plt.title("Forma de onda do áudio")
            plt.xlabel("Amostras")
            plt.ylabel("Amplitude")
            st.pyplot(plt.gcf())

        with st.spinner("🔍 Transcrevendo com Whisper..."):
            if diariazacao:
                diarized_segments = whisper_diarize(
                    audio_path,
                    num_speakers=num_speakers,
                    model_size=model_size,
                    language=language,
                )
                for seg in diarized_segments:
                    conversation.append(
                        {"speaker": seg["speaker"], "text": seg["text"].strip()}
                    )
            else:
                whisper_model = whisper.load_model(model_size)
                result = whisper_model.transcribe(audio_path, language=language)
                conversation.append(
                    {"speaker": "Speaker 1", "text": result["text"].strip()}
                )

elif input_mode == "Arquivo de Transcrição (.json)":
    uploaded_json = st.file_uploader(
        "📄 Envie o arquivo de transcrição (.json)", type=["json"]
    )
    if uploaded_json:
        conversation = json.load(uploaded_json)

elif input_mode == "Exemplo Interno":
    conversation = [
        {"speaker": "Cliente", "text": "Oi, estou com um problema na minha conta."},
        {"speaker": "Atendente", "text": "Claro, posso verificar isso para você."},
        {"speaker": "Cliente", "text": "Quero cancelar o serviço."},
    ]

# ------------------- EXIBIÇÃO ----------------------
if conversation:
    st.subheader("💬 Conversa")
    for turn in conversation:
        # Define cores distintas para até 5 falantes
        speaker_colors = {
            "Speaker 1": "#e1ffc7",
            "Speaker 2": "#d2e3fc",
            "Speaker 3": "#ffe0e0",
            "Speaker 4": "#f0e68c",
            "Speaker 5": "#d1c4e9",
        }
        speaker = turn["speaker"]
        bubble_color = speaker_colors.get(speaker, "#eeeeee")
        alignment = "flex-start" if speaker.endswith("1") else "flex-end"
        sentiment_score = ""

        with st.spinner("📊 Analisando sentimento..."):
            if sentiment_choice == "spaCy local":
                doc = nlp_sentiment(turn["text"])
                sentiment = doc.cats
                label = max(sentiment, key=sentiment.get)
                score = sentiment[label]
                sentiment_score = (
                    f"<br><small>Sentimento: {label} ({score*100:.1f}%)</small>"
                )
            else:
                sentiment_model = pipeline(
                    "text-classification", model="pysentimiento/bertweet-pt-sentiment"
                )
                sentiment_result = sentiment_model(turn["text"])[0]
                sentiment_score = f"<br><small>Sentimento: {sentiment_result['label']} ({sentiment_result['score']*100:.1f}%)</small>"

        st.markdown(
            f"""
            <div style='display: flex; justify-content: {alignment}; padding: 4px 0;'>
                <div style='background-color: {bubble_color}; padding: 10px 16px; border-radius: 12px; max-width: 70%;'>
                    <strong>{turn['speaker']}</strong><br>{turn['text']}{sentiment_score}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    full_transcript = " ".join([turn["text"] for turn in conversation])

    # st.subheader("🎯 Predição de Intenção")
    # with st.spinner("🎯 Classificando intenção..."):
    #     intent_result = intent_model(full_transcript)[0]
    #     st.write(
    #         f"**Intenção prevista:** {intent_result['label']} ({intent_result['score']*100:.1f}%)"
    #     )

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
