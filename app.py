import os
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "chatbot_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))