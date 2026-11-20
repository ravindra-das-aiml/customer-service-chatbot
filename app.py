import re
import json
import random
import pickle
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("Customer Service Chatbot 🤖")

FALLBACK = "Sorry, I didn't understand that. Can you rephrase?"
MIN_KNOWN_RATIO = 0.3       # at least 60% of the meaningful words must be known
CONFIDENCE_THRESHOLD = 0.4


@st.cache_resource
def load_files():
    model = pickle.load(open("chatbot_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    with open("intents.json", encoding="utf-8") as f:
        intents = json.load(f)
    return model, vectorizer, intents


model, vectorizer, intents = load_files()


def is_understandable(text):
    """Check meaningful (non stop-word) words against the training vocabulary."""
    words = re.findall(r"\b\w\w+\b", text.lower())
    if not words:
        return False
    content = [w for w in words if w not in ENGLISH_STOP_WORDS]
    if not content:          # only common words like "what can you do"
        return True
    known = [w for w in content if w in vectorizer.vocabulary_]
    return len(known) / len(content) >= MIN_KNOWN_RATIO


def get_response(text):
    if not is_understandable(text):
        return FALLBACK

    X = vectorizer.transform([text])
    if model.predict_proba(X).max() < CONFIDENCE_THRESHOLD:
        return FALLBACK

    tag = model.predict(X)[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return FALLBACK


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
