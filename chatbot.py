# chatbot.py

import streamlit as st
import openai
import tempfile
import os
from doc_uploader import handle_file_upload
from voice_io import transcribe_audio, record_audio, text_to_speech
from feedback import save_feedback
from db_memory import load_memory, save_memory
from summarizer import summarize_conversation
from translator import translate_text
from user_profile import load_user_profile, update_user_profile
from web_search import search_web
from vector_store import load_vector_store, store_document
from multimodel import ask_multimodel

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Chatbot", layout="wide")
st.title("🤖 Smart Streamlit Chatbot")

# --- SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'memory' not in st.session_state:
    st.session_state.memory = load_memory()

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = load_user_profile()

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# --- SIDEBAR SETTINGS ---
st.sidebar.title("🧠 Settings")
st.session_state.lang = st.sidebar.selectbox("Select Language", ["en", "hi", "mr"])
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])

# --- CHAT FUNCTION ---
def generate_response(user_input):
    translated_input = translate_text(user_input, to_lang='en')
    response = ask_multimodel(translated_input, model=model_choice)
    translated_output = translate_text(response, to_lang=st.session_state.lang)
    return translated_output, response

# --- INPUT AREA ---
with st.form(key='chat_form'):
    user_input = st.text_input("Type your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    translated_output, original_output = generate_response(user_input)
    st.session_state.chat_history.append((user_input, translated_output))
    save_memory(st.session_state.chat_history)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(translated_output)

# --- VOICE INPUT ---
st.markdown("---")
if st.button("🎤 Speak"):
    audio_file = record_audio()
    if audio_file:
        text = transcribe_audio(audio_file)
        st.text_input("Transcribed Text", value=text)
        translated_output, _ = generate_response(text)
        st.chat_message("user").write(text)
        st.chat_message("assistant").write(translated_output)
        text_to_speech(translated_output)

# --- DOCUMENT UPLOAD ---
st.markdown("### 📄 Upload Document")
uploaded_file = st.file_uploader("Upload file", type=['pdf', 'txt', 'docx'])
if uploaded_file:
    doc_text = handle_file_upload(uploaded_file)
    store_document(doc_text)
    st.success("Document stored in vector DB")

# --- SUMMARIZATION ---
if st.button("📝 Summarize Conversation"):
    summary = summarize_conversation(st.session_state.chat_history)
    st.info(summary)

# --- WEB SEARCH ---
st.markdown("---")
query = st.text_input("🔎 Search Web")
if st.button("Search") and query:
    results = search_web(query)
    st.write(results)

# --- FEEDBACK ---
st.markdown("---")
with st.form("feedback_form"):
    feedback = st.text_input("👍👎 Leave feedback")
    if st.form_submit_button("Submit Feedback"):
        save_feedback(feedback)
        st.success("Thanks for your feedback!")

# --- USER PROFILE ---
st.sidebar.markdown("---")
user_name = st.sidebar.text_input("Name", st.session_state.user_profile.get("name", ""))
if st.sidebar.button("Update Profile"):
    update_user_profile({"name": user_name})
    st.success("Profile updated")
