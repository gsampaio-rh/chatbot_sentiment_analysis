import streamlit as st
import requests
import pytesseract
from PIL import Image
import whisper
import os
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO
from pydub import AudioSegment
from langchain.llms import Ollama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# ---- Helper: List Ollama Models ----
def list_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception as e:
        return [f"Error: {e}"]


# ---- Load CardiffNLP Sentiment Model ----
sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
labels = ["negative", "neutral", "positive"]


def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    sentiment = labels[torch.argmax(probs)]
    return f"{sentiment.capitalize()} ({probs.max().item():.2f})"


# ---- Streamlit Layout ----
st.set_page_config(page_title="Modular AI Chatbot", layout="wide")
st.title("🧠 Modular AI Chatbot with LangGraph + Ollama")
st.markdown("Interact with a chatbot that branches into multi-modal analysis.")

# ---- Model Selection ----
available_models = list_ollama_models()

if available_models and not available_models[0].startswith("Error"):
    chat_model_name = st.sidebar.selectbox(
        "Select Chat Model", available_models, index=0
    )
else:
    st.sidebar.warning("Could not fetch models from Ollama.")
    chat_model_name = "llama3"

# ---- Initialize Models ----
chat_model = Ollama(model=chat_model_name)
ocr_model = pytesseract  # using Tesseract locally
transcriber = whisper.load_model("base")  # Whisper for audio transcription

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Chat UI ----
st.subheader("💬 Chat")
user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    with st.spinner("Bot is thinking..."):
        bot_response = chat_model(user_input)
        sentiment = analyze_sentiment(user_input)
        st.session_state.chat_history.append(
            {"role": "bot", "text": bot_response, "sentiment": sentiment}
        )

# ---- Display Chat ----
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["text"])
    elif message["role"] == "bot":
        with st.chat_message("assistant"):
            st.markdown(message["text"])
            st.caption(f"Sentiment: {message['sentiment']}")

# ---- File Uploads ----
st.sidebar.header("Upload Files")
img_file = st.sidebar.file_uploader("Upload Image (OCR)", type=["png", "jpg", "jpeg"])
audio_file = st.sidebar.file_uploader(
    "Upload Audio (Transcribe & Emotion)", type=["mp3", "wav"]
)

# ---- OCR ----
if img_file is not None:
    st.subheader("🖼️ OCR Result")
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Extracting text..."):
        text = ocr_model.image_to_string(image)
        st.text_area("Extracted Text", text)

# ---- Audio Transcription ----
if audio_file is not None:
    st.subheader("🔊 Audio Analysis")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        audio_bytes = audio_file.read()
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    # Transcription
    with st.spinner("Transcribing audio..."):
        result = transcriber.transcribe(tmp_audio_path)
        st.text_area("Transcript", result["text"])

    # Emotion/Frequency Analysis (Simple Placeholder)
    with st.spinner("Analyzing Emotion..."):
        audio = AudioSegment.from_file(tmp_audio_path)
        loudness = audio.dBFS
        if loudness > -20:
            emotion = "😠 Loud / Possibly Angry"
        elif loudness > -35:
            emotion = "😐 Neutral"
        else:
            emotion = "😊 Calm / Quiet"
        st.markdown(f"**Emotion Estimate:** {emotion}")

# ---- Timeline Visual ----
# st.markdown("---")
# st.subheader("📊 Process Flow")
# steps = ["User Input", "Chatbot", "Sentiment", "OCR", "Audio Transcript", "Emotion"]
# fig, ax = plt.subplots(figsize=(10, 1))
# ax.axis("off")
# for i, step in enumerate(steps):
#     ax.text(
#         i,
#         0,
#         f"{i+1}. {step}",
#         ha="center",
#         va="center",
#         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black"),
#     )
# plt.xlim(-1, len(steps))
# st.pyplot(fig)

# st.markdown("---")
# st.caption(
#     "Demo powered by Streamlit, LangChain, LangGraph, Ollama, Whisper, Tesseract, and HuggingFace Transformers."
# )
