import streamlit as st
from infrance_model import inf_model
import pandas as pd
from Vector_DB import vector_db
from whisper_model import WhisperModel
from summarization import Summarizer

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
db = vector_db()

st.sidebar.header('MeetingQA & Summarizer')
st.sidebar.write("")
st.sidebar.write('')

sidebaroption = st.sidebar.selectbox('Select Data Type:', ('YouTube Link', 'Document', 'Audio File'))

if sidebaroption == 'YouTube Link':
    video_link = st.sidebar.text_input("Please Enter YouTube Link")
    
    if video_link:
        wh_model = WhisperModel()
        audio = wh_model.download(video_link)
        audio_text = wh_model.trans("audio.mp3.mp3")
        dataframe = pd.DataFrame({'Transcript': [audio_text]})
        db.create_vector_db(dataframe['Transcript'][0])
        st.sidebar.write("Vector DB created successfully")

if sidebaroption == 'Document':
    uploaded_file = st.sidebar.file_uploader("Choose a text file", type="txt")
    if uploaded_file is not None:
        try:
            text_data = uploaded_file.read().decode("utf-8")
            st.sidebar.write("Text file loaded successfully")
            dataframe = pd.DataFrame({'Transcript': [text_data]})
            db.create_vector_db(dataframe['Transcript'][0])
            st.sidebar.write("Vector DB created successfully")
        except Exception as e:
            st.write(f"An error occurred while reading the TXT file: {e}")
    else:
        st.write("Please upload a TXT file.")

if sidebaroption == 'Audio File':
    audio_file = st.sidebar.file_uploader("Upload Audio File", type=["mp4", "mkv", "mov", "mp3"])
    if audio_file is not None:
        wh_model = WhisperModel()
        audio_text = wh_model.trans(audio_file)
        dataframe = pd.DataFrame({'Transcript': [audio_text]})
        db.create_vector_db(dataframe['Transcript'][0])
        st.sidebar.write("Vector DB created successfully")

st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Layout for chat input and summarization button


if prompt := st.chat_input():
    model = inf_model()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    transcript = db.query_vector_db(prompt)
    response = model.predict(prompt, transcript)
    response = response.split("### Response:")
    msg = response[1]
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


if st.sidebar.button("Summarize Meeting"):
    model = inf_model()
    Summarizer = Summarizer()
    chunks = db.read_all_chunks()
    summary = Summarizer.summarize(chunks)

    llm_summary = model.llm_summarize(summary)

    st.session_state.messages.append({"role": "assistant", "content": "Summarized Meeting"})

    llm_summary = llm_summary.split("### Response:")

    llm_summary = llm_summary[1]

    st.session_state.messages.append({"role": "assistant", "content": llm_summary})
    
    st.chat_message("assistant").write(llm_summary)