import streamlit as st
import librosa
import numpy as np
import torch
import torchaudio
import whisper
import tempfile
import plotly.graph_objects as go
from transformers import pipeline

# Load Whisper model once
whisper_model = whisper.load_model("base")

# Placeholder models (replace with your fine-tuned versions)
sentiment_model = pipeline("sentiment-analysis")
intent_model = pipeline("text-classification", model="bert-base-uncased")

st.set_page_config(page_title="Voice Intelligence App", layout="centered")
st.title("🎙️ Voice Intelligence App")
st.markdown(
    "Upload uma conversa em áudio para análise automatizada de sentimento e intenção."
)

# Upload de áudio
uploaded_file = st.file_uploader("🔊 Envie um arquivo de áudio", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Salvar temporariamente o arquivo de áudio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Carregar áudio para onda
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
    st.subheader("✍️ Transcrição")
    result = whisper_model.transcribe(tmp_path)
    transcript = result["text"]
    st.markdown(f"> {transcript}")

    # Análise de Sentimento
    st.subheader("❤️ Análise de Sentimento")
    sentiment_result = sentiment_model(transcript)[0]
    st.write(
        f"**Sentimento:** {sentiment_result['label']} ({sentiment_result['score']*100:.1f}%)"
    )

    # Predição de Intenção
    st.subheader("🎯 Predição de Intenção")
    intent_result = intent_model(transcript)[0]
    st.write(
        f"**Intenção prevista:** {intent_result['label']} ({intent_result['score']*100:.1f}%)"
    )

    # Estilo Apple-like
    st.markdown(
        """
    <style>
        .stApp {
            background-color: #f9f9f9;
            color: #2c2c2c;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 16px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
