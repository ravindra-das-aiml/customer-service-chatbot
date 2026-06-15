import streamlit as st
import json
import pickle
import random

model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

st.title("Customer Service Chatbot 🤖")

user_input = st.text_input("Ask something")

if user_input:
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            st.success(random.choice(intent["responses"]))