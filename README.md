<div align="center">

# 🤖 MeetingQA

### AI-Powered Meeting Assistant with Fine-Tuned LLM

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/MahmoudMohamed)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![MeetingQA Banner](https://github.com/m-mahmoud-mohamed/MeetingQA/assets/117641794/57b497d2-be4e-4b8c-b119-37c615f9b469)

</div>

---

## 📖 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Architecture](#-architecture)
- [Pipeline](#-pipeline)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Contributing](#-contributing)

---

## 🧠 About

**MeetingQA** is an intelligent meeting assistant that lets you interact with your meetings through natural language. Upload a YouTube video, audio recording, or text document and instantly unlock two powerful capabilities:

- 💬 **Ask questions** about the meeting content and get accurate, context-aware answers.
- 📝 **Generate summaries** that distill the key points from long recordings or transcripts.

The system is powered by a fine-tuned **Phi-3** model (trained with Direct Preference Optimization) and uses a **Retrieval-Augmented Generation (RAG)** pipeline backed by Weaviate vector database.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 YouTube Support | Paste a YouTube URL to transcribe and analyze the video |
| 🎙️ Audio File Upload | Upload MP3, MP4, MKV, or MOV audio/video files |
| 📄 Document Upload | Upload plain-text (`.txt`) documents |
| 💬 Contextual Q&A | Ask any question about your meeting content |
| 📋 Meeting Summarization | Auto-generate concise meeting summaries |
| 🔍 Semantic Search | RAG pipeline retrieves the most relevant context for each query |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Input                            │
│           (YouTube Link / Audio File / Document)             │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Transcription (Whisper) │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Semantic Chunking (LangChain) │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
 ┌─────────────────────┐   ┌────────────────────────┐
 │ Embedding           │   │ Summarization          │
 │ (BAAI/bge-base-en)  │   │ (Fine-tuned T5)        │
 └──────────┬──────────┘   └────────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ Weaviate Vector DB  │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────────────────────┐
 │  Fine-Tuned Phi-3 (DPO)            │
 │  Q&A  │  Meeting Summary           │
 └─────────────────────────────────────┘
```

---

## 🔄 Pipeline

### Step-by-Step Flow

1. **File Upload** — The user provides a YouTube link, uploads an audio/video file, or a `.txt` document via the Streamlit sidebar.

2. **Transcription** — Audio content is transcribed to text using [OpenAI Whisper](https://github.com/openai/whisper) (`small` model).

3. **Semantic Chunking** — The transcript is split into semantically coherent chunks using LangChain's `SemanticChunker`, then further constrained to 500-token sub-chunks using the T5 tokenizer.

4. **Embedding** — Each chunk is embedded using `BAAI/bge-base-en-v1.5` via FastEmbed.

5. **Vector Database Storage** — Embeddings and their corresponding text chunks are stored in a [Weaviate](https://weaviate.io/) vector database for fast semantic retrieval.

6. **Chat / Summarization**
   - **Q&A**: The user's query is embedded, the top-4 most relevant chunks are retrieved, and the fine-tuned Phi-3 model generates a grounded answer.
   - **Summarization**: All chunks are passed through a fine-tuned T5 summarization model, then the Phi-3 model produces a polished final summary.

---

## 🤗 Models

| Model | Role | Link |
|---|---|---|
| `MahmoudMohamed/Phi3_DPO_MeetingQA_4bit` | Chat & Summarization LLM (fine-tuned Phi-3 with DPO) | [Hugging Face](https://huggingface.co/MahmoudMohamed/Phi3_DPO_MeetingQA_4bit) |
| `Falconsai/text_summarization` | Chunk-level Summarization (fine-tuned T5) | [Hugging Face](https://huggingface.co/Falconsai/text_summarization) |
| `BAAI/bge-base-en-v1.5` | Text Embeddings | [Hugging Face](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| `openai/whisper-small` | Audio Transcription | [GitHub](https://github.com/openai/whisper) |

The Phi-3 model was fine-tuned using:
- **Supervised Fine-Tuning (SFT)** — See [`NoteBooks/Phi3_finetuning.ipynb`](NoteBooks/Phi3_finetuning.ipynb)
- **Direct Preference Optimization (DPO)** — See [`NoteBooks/DPO_Phi3_finetuning.ipynb`](NoteBooks/DPO_Phi3_finetuning.ipynb)
- **Dataset Preprocessing** — See [`NoteBooks/Dataset_preprocessing.ipynb`](NoteBooks/Dataset_preprocessing.ipynb)

---

## 📁 Project Structure

```
MeetingQA/
├── Meeting.py              # Streamlit app entry point
├── infrance_model.py       # Phi-3 inference (Q&A & summarization)
├── whisper_model.py        # Audio transcription & YouTube download
├── summarization.py        # T5-based chunk summarization
├── Vector_DB.py            # Weaviate vector DB wrapper
├── requirements.txt        # Python dependencies
├── NoteBooks/
│   ├── Phi3_finetuning.ipynb         # SFT fine-tuning notebook
│   ├── DPO_Phi3_finetuning.ipynb     # DPO fine-tuning notebook
│   └── Dataset_preprocessing.ipynb  # Dataset preparation
└── Evaluation/
    ├── TIGERScoreEval.ipynb  # Model evaluation with TIGERScore
    ├── eval-data.json        # Evaluation dataset
    └── test.ipynb            # Additional tests
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Weaviate](https://weaviate.io/) cloud instance (or local)
- GPU recommended for faster inference (CPU is supported)
- [FFmpeg](https://ffmpeg.org/) installed for audio processing

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/m-mahmoud-mohamed/MeetingQA.git
   cd MeetingQA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Weaviate:**

   Update the Weaviate endpoint and API key in `Vector_DB.py`:
   ```python
   self.client = weaviate.Client(
       url="YOUR_WEAVIATE_URL",
       auth_client_secret=weaviate.auth.AuthApiKey(api_key="YOUR_API_KEY"),
   )
   ```

---

## 💡 Usage

### 1. Launch the App

```bash
streamlit run Meeting.py
```

### 2. Provide Your Meeting Content

Open the app in your browser and select one of the input modes from the sidebar:

| Mode | How to Use |
|---|---|
| **YouTube Link** | Paste a YouTube video URL into the text box |
| **Audio File** | Upload an MP3, MP4, MKV, or MOV file |
| **Document** | Upload a plain `.txt` transcript |

The system will automatically transcribe, chunk, embed, and index the content.

### 3. Interact with Your Meeting

- **Chat:** Type any question in the chat input box to get an AI-powered answer grounded in the meeting content.
- **Summarize:** Click the **"Summarize Meeting"** button in the sidebar to generate a comprehensive summary.

---

## 📊 Evaluation

Model quality is assessed using **TIGERScore**, an explainable reference-free metric for evaluating text generation. Evaluation notebooks and data are located in the [`Evaluation/`](Evaluation/) directory.

---

