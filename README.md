# Meeting Chat Bot Based on Fine-Tuned LLM
![WhatsApp Image 2024-05-23 at 02 13 30](https://github.com/m-mahmoud-mohamed/MeetingQA/assets/117641794/57b497d2-be4e-4b8c-b119-37c615f9b469)

## About the Project
Our project focuses on developing a meeting chat bot that utilizes a fine-tuned large language model (LLM) to assist users in interacting with their video, audio, or document files. The bot can answer questions about the uploaded content and provide concise summaries. The main components of the project include transcription, embedding, vector database storage, summarization, and a chat interface.

## Project Pipeline

## Overview
1-Upload File: User uploads a file (video, audio, or document) through the bot's user interface.
2-Transcription: The file is converted into a transcript using Whisper.
3-Embedding: The transcript is embedded using an embedding model.
5-Vector Database Storage: The embedding results are stored in a vector database (Weaviate-DB) and split into chunks.
6-Summarization: Each chunk is passed through a summarization model (fine-tuned T5) and both the original and summarized content are stored for retrieval.
7-Chat Interaction: The user can either chat with the bot about the uploaded file or request a summary of the file's content. The bot uses a fine-tuned LLM (Phi-3) to generate responses.

## Detailed Steps
### 1-File Upload:
The user uploads their file (video, audio, or document) through the UI.
### 2-Transcription:
The uploaded file is transcribed into text using Whisper.
### 3-Embedding:
The transcript is processed by an embedding model to generate embeddings.
### 4-Vector Database Storage:
The embeddings are stored in a vector database (Weaviate-DB) and split into manageable chunks.
Both the summarized and original content are stored for retrieval.
### 5-Summarization:
The chunks are fed into a summarization model (fine-tuned T5) to produce a summary.
The summarized content and the original content are stored for future reference.
### 6-Chat Interaction:

The user can chat with the bot in the context of their uploaded file or request a summary.
The bot utilizes a fine-tuned LLM (Phi-3) to generate accurate responses based on the context of the uploaded file.

## How to Use the Bot
### 1-Install Requirements :
```console
pip install -r requirements.txt
```
### 2-Run Program:
```console
streamlit run Meeting.py
```

### 3-Upload File:
Upload your file (video, audio, or document) through the system's UI.
### 4-Interact with the Bot:
Choose to either chat with the bot about your uploaded file or ask the bot to summarize the file's content.


