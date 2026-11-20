# 🤖 Customer Service Chatbot

An AI-powered customer service chatbot built using **Python, scikit-learn and Streamlit**. The bot understands a user's message, classifies its intent using a trained machine learning model, and replies with a suitable response.

> Developed as part of a 1-month internship at **Codec Technologies**.

---

## 📌 Features

- Intent classification using a trained ML model (TF-IDF + scikit-learn)
- Predefined intents and responses stored in `intents.json` (easy to extend)
- Simple and clean web interface built with Streamlit
- Pre-trained model and vectorizer included, so it runs without retraining

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| ML Library | scikit-learn |
| Text Processing | TF-IDF Vectorizer |
| Web App | Streamlit |
| Model Storage | Pickle (`.pkl`) |

---

## 📂 Project Structure

```
customer-service-chatbot/
├── .devcontainer/       # Dev container configuration
├── app.py               # Streamlit application
├── train.py             # Script to train the model
├── intents.json         # Intents, patterns and responses
├── chatbot_model.pkl    # Trained classification model
├── vectorizer.pkl       # Trained TF-IDF vectorizer
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## ⚙️ How It Works

1. Intents and sample patterns are defined in `intents.json`.
2. `train.py` converts the text into numerical features using TF-IDF and trains a classifier.
3. The trained model and vectorizer are saved as `chatbot_model.pkl` and `vectorizer.pkl`.
4. `app.py` loads them, predicts the intent of the user's message, and shows the matching response.

---

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/ravindra-das-aiml/customer-service-chatbot.git
cd customer-service-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Retrain the model**
```bash
python train.py
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📸 Screenshots

_Add screenshots of the app here._

```
![App Screenshot](screenshots/app.png)
```

---

## 🔮 Future Improvements

- Add more intents and training data
- Use deep learning / transformer-based models for better accuracy
- Add chat history and multi-turn conversations
- Deploy on Streamlit Community Cloud

---

## 🎓 Internship

This project was developed during a 1-month internship at **Codec Technologies**.

## 👤 Author

**Ravindra Das**
GitHub: [@ravindra-das-aiml](https://github.com/ravindra-das-aiml)
