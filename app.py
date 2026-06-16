import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("Free Smart Chatbot 🤖 (No API)")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

user_input = st.text_input("Ask anything:")

if user_input:

    new_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    if st.session_state.chat_history is not None:
        bot_input = torch.cat([st.session_state.chat_history, new_input], dim=-1)
    else:
        bot_input = new_input

    chat_history = model.generate(
        bot_input,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    st.session_state.chat_history = chat_history

    reply = tokenizer.decode(
        chat_history[:, bot_input.shape[-1]:][0],
        skip_special_tokens=True
    )

    st.write("Bot:", reply)