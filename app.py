import json
import random
import pickle
import streamlit as st

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("Customer Service Chatbot 🤖")

@st.cache_resource
def load_files():
    model = pickle.load(open("chatbot_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    with open("intents.json", encoding="utf-8") as f:
        intents = json.load(f)
    return model, vectorizer, intents

model, vectorizer, intents = load_files()

def get_response(text):
    tag = model.predict(vectorizer.transform([text]))[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

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
