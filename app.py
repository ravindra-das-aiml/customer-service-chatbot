import re
import json
import random
import pickle
import streamlit as st

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("Customer Service Chatbot 🤖")

FALLBACK = "Sorry, I didn't understand that. Can you rephrase?"
MIN_KNOWN_WORDS = 0.6      # at least 60% of the words must be known to the model
CONFIDENCE_THRESHOLD = 0.4


@st.cache_resource
def load_files():
    model = pickle.load(open("chatbot_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    with open("intents.json", encoding="utf-8") as f:
        intents = json.load(f)
    return model, vectorizer, intents


model, vectorizer, intents = load_files()


def is_known_text(text):
    """Return True if most words of the message exist in the training vocabulary."""
    words = re.findall(r"\b\w\w+\b", text.lower())
    if not words:
        return False
    vocab = getattr(vectorizer, "vocabulary_", {})
    known = [w for w in words if w in vocab]
    return len(known) / len(words) >= MIN_KNOWN_WORDS


def get_response(text):
    # Unrelated question (most words are unknown) -> fallback
    if not is_known_text(text):
        return FALLBACK

    X = vectorizer.transform([text])

    # Low model confidence -> fallback
    if hasattr(model, "predict_proba"):
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
