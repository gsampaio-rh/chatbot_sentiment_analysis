import streamlit as st
import librosa
import numpy as np
import torch
import torchaudio
import whisper
import tempfile
import plotly.graph_objects as go
import spacy
import json
from transformers import pipeline

# Load Whisper model once
whisper_model = whisper.load_model("base")

# Load spaCy model
nlp_sentiment = spacy.load("spacy_model/model-best")

# Placeholder for intent model
intent_model = pipeline("text-classification", model="bert-base-uncased")

st.set_page_config(page_title="Voice Intelligence App", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurações")
    st.markdown("Configure os parâmetros de entrada e análise")
    input_mode = st.radio(
        "Tipo de Entrada:",
        ("MP3/WAV", "Arquivo de Transcrição (.json)", "Exemplo Interno"),
    )
    sentiment_choice = st.radio(
        "Modelo de Sentimento:", ("spaCy local", "Transformers (online)")
    )
    diariazacao = st.checkbox("Realizar diarização de falantes", value=True)

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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Visualização da onda sonora
        y, sr = librosa.load(tmp_path, sr=None)
        audio = y[::100]
        time_axis = np.linspace(0, len(y) / sr, num=len(audio))
        colors = np.interp(
            np.abs(audio), (np.min(np.abs(audio)), np.max(np.abs(audio))), (0, 1)
        )
        colors = [
            f"rgba({int(255*c)}, {int(100 + 155*(1-c))}, {int(255*(1-c))}, 0.9)"
            for c in colors
        ]

        fig = go.Figure()
        for i in range(1, len(audio)):
            fig.add_trace(
                go.Scatter(
                    x=[time_axis[i - 1], time_axis[i]],
                    y=[audio[i - 1], audio[i]],
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="🎧 Onda Sonora Interativa e Colorida",
            xaxis_title="Tempo (s)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Transcrição com Whisper
        result = whisper_model.transcribe(
            tmp_path, condition_on_previous_text=True, fp16=False
        )
        transcript = result["text"]
        if diariazacao and "segments" in result:
            for segment in result["segments"]:
                speaker = (
                    f"Speaker {segment['speaker'] if 'speaker' in segment else '1'}"
                )
                conversation.append(
                    {"speaker": speaker, "text": segment["text"].strip()}
                )
        else:
            conversation.append({"speaker": "Speaker 1", "text": transcript.strip()})

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

# Exibição da conversa estilo WhatsApp
if conversation:
    st.subheader("💬 Conversa")
    for turn in conversation:
        alignment = "flex-start" if "cliente" in turn["speaker"].lower() else "flex-end"
        bubble_color = "#e1ffc7" if "cliente" in turn["speaker"].lower() else "#d2e3fc"
        sentiment_score = ""

        if sentiment_choice == "spaCy local":
            doc = nlp_sentiment(turn["text"])
            sentiment = doc.cats
            label = max(sentiment, key=sentiment.get)
            score = sentiment[label]
            sentiment_score = (
                f"<br><small>Sentimento: {label} ({score*100:.1f}%)</small>"
            )
        else:
            # sentiment_model = pipeline("sentiment-analysis")
            # Diretamente do Hugging Face
            sentiment_model = pipeline("text-classification", model="pysentimiento/bertweet-pt-sentiment")

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

    # Predição de Intenção
    st.subheader("🎯 Predição de Intenção")
    intent_result = intent_model(full_transcript)[0]
    st.write(
        f"**Intenção prevista:** {intent_result['label']} ({intent_result['score']*100:.1f}%)"
    )

# Estilo Apple-like com ajustes mais suaves
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
