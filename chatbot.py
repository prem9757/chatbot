import streamlit as st
import os
import tempfile
import time
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
from deep_translator import GoogleTranslator
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# --- Voice Input ---
def transcribe_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError as e:
        return f"❌ Speech Recognition error: {str(e)}"

# --- Voice Output ---
def speak(text):
    tts = gTTS(text)
    fp = BytesIO()
    tts.write_to_fp(fp)
    st.audio(fp.getvalue(), format='audio/mp3')

# --- Translation ---
def translate_text(text, target_lang="en"):
    try:
        return GoogleTranslator(target=target_lang).translate(text)
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# --- Summarizer ---
def summarize_text(docs):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

# --- Vector DB Storage ---
def store_file_to_vector_db(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# --- Document Upload ---
def handle_file_upload():
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])
    if uploaded_file:
        st.success("✅ File uploaded successfully.")
        return store_file_to_vector_db(uploaded_file)
    return None

# --- Feedback ---
def collect_feedback():
    st.markdown("### 🙋 Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback! 😊")
    with col2:
        if st.button("👎 No"):
            st.warning("We'll try to do better! 🙏")

# --- User Profile ---
def get_user_profile():
    with st.sidebar:
        st.header("🧑‍💼 User Profile")
        name = st.text_input("Name")
        lang = st.selectbox("Preferred Language", ["en", "hi", "mr", "ta", "te"])
        voice = st.toggle("🎤 Enable Voice Input/Output")
        return name, lang, voice

# --- Main App ---
def main():
    st.set_page_config(page_title="Smart Chatbot", layout="centered")
    st.title("🤖 Smart Multilingual Chatbot with Voice & Memory")

    name, lang, voice_enabled = get_user_profile()
    vector_db = handle_file_upload()

    st.markdown("---")
    st.subheader("💬 Ask a question")

    if voice_enabled:
        if st.button("🎙️ Speak"):
            user_input = transcribe_voice()
            st.text_area("🎧 Recognized Text", value=user_input, height=100)
        else:
            user_input = st.text_input("Type your message")
    else:
        user_input = st.text_input("Type your message")

    if user_input:
        with st.spinner("Thinking..."):
            translated = translate_text(user_input, target_lang="en")

            if vector_db:
                llm = OpenAI(temperature=0)
                qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever())
                response = qa_chain.run(translated)
            else:
                llm = OpenAI(temperature=0)
                response = llm(translated)

            final_output = translate_text(response, target_lang=lang)
            st.success(final_output)

            if voice_enabled:
                speak(final_output)

            collect_feedback()

if __name__ == "__main__":
    main()
