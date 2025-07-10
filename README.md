# 🧠 Voice Intelligence App – Transcrição, Sentimento e Intenção em Português

Este repositório fornece um conjunto completo de ferramentas para processar, transcrever e analisar sentimentos e intenções de chamadas telefônicas ou arquivos de áudio em **português**. Ele combina modelos avançados como **Whisper**, **BERT** e **spaCy** com uma interface interativa via **Streamlit**.

---

## 📦 Funcionalidades

- 🎙️ **Transcrição de Áudio** com modelos `faster-whisper`
- 😊 **Análise de Sentimento** via `spaCy` ou modelos da HuggingFace
- 🎯 **Detecção de Intenção** com BERT fine-tunado em chamadas reais
- 💻 **Interface Streamlit** para uso interativo
- 📊 **Visualizações** integradas (gráficos, distribuições, barras)
- 🧪 **Notebooks** para fine-tuning, avaliação e pré-processamento

---

## 🗂 Estrutura do Repositório

```text

├── streamlit-audio.ipynb         # Interface principal (Streamlit)
├── 1_sentiment_analysis_bert.ipynb   # Treinamento de modelo BERT para sentimento
├── 1_sentiment_analysis_spacy.ipynb  # Treinamento de modelo spaCy para sentimento
├── 2_batch_transcript.ipynb     # Transcrição em lote de áudios
├── 3_intent_analysis.ipynb      # Classificação de intenção com LLaMA + BERT
├── dataset/                     # Datasets processados
├── transcripts/                 # Arquivos de transcrição (.json)
├── fine\_tuned\_intent\_bert/     # Modelo de intenção treinado
├── spacy\_model/                # Modelo de sentimento (spaCy) treinado
├── audios/                      # Arquivos de entrada de áudio (.mp3, .wav)
├── requirements.txt             # Dependências
└── README.md
```

---

## 🚀 Executando o App (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit-audio.ipynb
````

> Ou converta o notebook para `.py` com:
> `jupyter nbconvert --to script streamlit-audio.ipynb && streamlit run streamlit-audio.py`

---

## 📁 Notebooks e Experimentos

| Notebook                           | Objetivo                                         |
| ---------------------------------- | ------------------------------------------------ |
| `1_sentiment_analysis_bert.ipynb`  | Treinamento de BERT com IMDB em português        |
| `1_sentiment_analysis_spacy.ipynb` | Treinamento leve com spaCy e tweets rotulados    |
| `2_batch-transcript.ipynb`         | Transcrição em lote de áudios com faster-whisper |
| `3_intent-analysis.ipynb`          | Classificação de intenção via LLaMA + BERT       |

---

## 📊 Modelos Utilizados

### 🎙 Transcrição – `faster-whisper`

* Modelos de `tiny` até `large-v3`, inclusive versões `distil` e `turbo`
* Compatível com CPU, CUDA, MPS (Apple Silicon)

### 😊 Análise de Sentimento

* `spaCy` local (`pt_core_news_lg`)
* `bertweet-pt-sentiment` da Hugging Face

### 🎯 Detecção de Intenção

* BERT fine-tunado com transcrições reais
* Suporta 7 categorias: `cancelamento`, `reclamação`, `elogio`, etc.

---

## 📈 Exemplo de Visualização

* Intenção e sentimentos são exibidos ao lado dos balões de transcrição.
* Gráficos:

  * Pizza: distribuição de sentimentos por frase
  * Barras: score de intenção + frequência de sentimentos

---

## 🧪 Requisitos

* Python 3.9+
* PyTorch com suporte a CUDA/MPS (opcional)
* Modelos locais:

  * `output_model/model-best/` (spaCy)
  * `fine_tuned_intent_bert/` (BERT)
* FFmpeg instalado para conversão de áudio

---

## 🛡️ Licença

Este projeto é distribuído sob a licença MIT.

---

## 🙋‍♂️ Contribuição

Sinta-se à vontade para abrir issues, pull requests ou sugerir melhorias. Feedbacks são bem-vindos!

---

## 🔖 Créditos

* [HuggingFace](https://huggingface.co/)
* [spaCy](https://spacy.io/)
* [faster-whisper](https://github.com/guillaumekln/faster-whisper)
* [Ollama](https://ollama.com) para LLaMA 3
