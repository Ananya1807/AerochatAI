"""
Streamlit Chatbot (Dialogflow-style)
Single-file app: streamlit_chatbot.py

Features:
- Intent classification (TF-IDF + LinearSVC)
- Simple entity extraction (regex-based for dates, origin/destination)
- Multi-turn dialogue manager with slot-filling
- Streamlit UI with chat history, diagnostics, retrain button

Run:
1) python -m venv venv && source venv/bin/activate   # or `venv\Scripts\activate` on Windows
2) pip install -U pip
3) pip install streamlit scikit-learn joblib
4) streamlit run streamlit_chatbot.py

Note: training runs fast for the tiny sample intents included. Expand intents_list below for richer behavior.
"""

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import random
from collections import defaultdict
import json
from pathlib import Path

# -----------------
# Intents spec
# -----------------
intents_spec = {
    "intents": [
        {
            "name": "greet",
            "examples": ["hi", "hello", "hey there", "good morning"],
            "responses": ["Hey! How can I help you today?", "Hi — what can I do for you?"]
        },
        {
            "name": "book_flight",
            "examples": [
                "I want to book a flight to Paris",
                "Book me a ticket to New York on 2025-12-20",
                "Need a flight from Mumbai to Dubai next Monday"
            ],
            "slots": ["origin", "destination", "date"],
            "responses": ["Sure — I can help book that. Where are you flying from?", "When do you want to travel?"]
        },
        {
            "name": "provide_date",
            "examples": ["I will travel on 2025-12-20", "next Monday", "tomorrow"],
            "responses": ["Got the date. Do you have a preferred time?"]
        },
        {
            "name": "goodbye",
            "examples": ["bye", "see you", "thanks bye"],
            "responses": ["Goodbye! Safe travels."]
        }
    ]
}

MODEL_PATH = Path("model_streamlit.joblib")

# -----------------
# Training utilities
# -----------------

def train_model(spec):
    texts, labels = [], []
    for it in spec["intents"]:
        for ex in it.get("examples", []):
            texts.append(ex)
            labels.append(it["name"])
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), LinearSVC())
    model.fit(texts, y)
    joblib.dump({"model": model, "label_enc": le, "spec": spec}, MODEL_PATH)
    return model, le


def load_or_train(spec):
    if MODEL_PATH.exists():
        art = joblib.load(MODEL_PATH)
        return art["model"], art["label_enc"]
    else:
        return train_model(spec)


# -----------------
# NLU: Intent + Entities
# -----------------

def predict_intent(model, le, text):
    try:
        pred = model.predict([text])[0]
        distances = None
        # attempt to get margins via decision_function
        try:
            dist = model.named_steps["linearsvc"].decision_function(model.named_steps["tfidf"].transform([text]))
            # dist shape (1, n_classes)
            if dist is not None:
                # apply softmax-ish to distances for probability proxy
                import math
                exps = [math.exp(d) for d in dist[0]]
                s = sum(exps)
                probs = [e / s for e in exps]
                conf = max(probs)
            else:
                conf = 0.8
        except Exception:
            conf = 0.82
        intent = le.inverse_transform([pred])[0]
        return {"intent": intent, "confidence": float(conf)}
    except Exception:
        return {"intent": "fallback", "confidence": 0.0}


def extract_entities(text):
    text = text.strip()
    ents = {}
    # date patterns: YYYY-MM-DD, today, tomorrow, next <word>
    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|today|tomorrow|next\s+\w+)", text, re.I)
    if date_match:
        ents.setdefault("date", []).append(date_match.group(0))
    # origin/destination: look for 'from X' and 'to Y'
    from_match = re.search(r"from\s+([A-Z][a-zA-Z0-9_\- ]+)", text)
    to_match = re.search(r"to\s+([A-Z][a-zA-Z0-9_\- ]+)", text)
    if from_match:
        v = from_match.group(1).strip()
        ents.setdefault("origin", []).append(v)
    if to_match:
        v = to_match.group(1).strip()
        ents.setdefault("destination", []).append(v)
    # fallback: if user writes 'Paris' or 'Mumbai' (single capitalized token)
    cap_tokens = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
    # simple heuristic: if a capitalized token appears and no dest, set as destination
    if cap_tokens and "destination" not in ents:
        ents.setdefault("destination", []).append(cap_tokens[-1])
    return ents


# -----------------
# Dialogue manager
# -----------------

class Conversation:
    def __init__(self, uid):
        self.uid = uid
        self.state = "idle"
        self.slots = defaultdict(lambda: None)
        self.history = []

class DialogueManager:
    def __init__(self, intents_spec):
        self.sessions = {}
        self.intents = {it["name"]: it for it in intents_spec["intents"]}

    def get_session(self, uid):
        if uid not in self.sessions:
            self.sessions[uid] = Conversation(uid)
        return self.sessions[uid]

    def handle(self, uid, intent_name, entities, text):
        s = self.get_session(uid)
        s.history.append({"intent": intent_name, "text": text, "entities": entities})
        # policy rules
        if intent_name == "greet":
            s.state = "idle"
            return random.choice(self.intents["greet"]["responses"]), s
        if intent_name == "book_flight":
            s.state = "collecting_slots"
            # fill slots
            for k, v in entities.items():
                if k in ("destination", "origin"):
                    s.slots[k] = v[0]
                if k == "date":
                    s.slots["date"] = v[0]
            missing = [slot for slot in self.intents["book_flight"].get("slots", []) if not s.slots.get(slot)]
            if missing:
                question = f"I still need: {', '.join(missing)}."
                return f"{question} {random.choice(self.intents['book_flight']['responses'])}", s
            else:
                s.state = "confirm"
                return f"Got it — book from {s.slots.get('origin','unknown')} to {s.slots.get('destination')} on {s.slots.get('date')}. Confirm?", s
        if s.state == "collecting_slots" or intent_name == "provide_date":
            for k, v in entities.items():
                if k == "date":
                    s.slots["date"] = v[0]
                if k == "destination":
                    s.slots["destination"] = v[0]
                if k == "origin":
                    s.slots["origin"] = v[0]
            missing = [slot for slot in self.intents["book_flight"].get("slots", []) if not s.slots.get(slot)]
            if missing:
                return f"Still missing: {', '.join(missing)}", s
            else:
                s.state = "confirm"
                return f"All set: origin {s.slots.get('origin')}, dest {s.slots.get('destination')}, date {s.slots.get('date')}. Confirm booking?", s
        if intent_name == "goodbye":
            s.state = "done"
            return random.choice(self.intents["goodbye"]["responses"]), s
        return "Sorry, I didn't understand. Can you rephrase?", s


# -----------------
# Streamlit UI
# -----------------

st.set_page_config(page_title="Dialogflow-style Chatbot", layout="wide")
st.title("📬 Dialogflow-style Chatbot — Streamlit")

# training / load
with st.sidebar:
    st.header("Model")
    if st.button("Retrain model"):
        with st.spinner("Retraining..."):
            model, le = train_model(intents_spec)
            st.success("Model retrained and saved.")
    else:
        model, le = load_or_train(intents_spec)
    st.write("Model file:", str(MODEL_PATH))
    st.markdown("---")
    st.header("Session")
    if "uid" not in st.session_state:
        st.session_state["uid"] = "local_user"
    st.text_input("User ID", key="uid")
    if st.button("Reset conversation"):
        if "dm" in st.session_state:
            del st.session_state["dm"]
        st.experimental_rerun()

# init dialogue manager in session state
if "dm" not in st.session_state:
    st.session_state["dm"] = DialogueManager(intents_spec)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("You:")
    if st.button("Send") and user_input:
        nlu = predict_intent(model, le, user_input)
        entities = extract_entities(user_input)
        reply, session = st.session_state["dm"].handle(st.session_state["uid"], nlu["intent"], entities, user_input)
        # append to history
        st.session_state["history"].append(("user", user_input))
        st.session_state["history"].append(("bot", reply))

    # display chat history
    for who, text in st.session_state["history"]:
        if who == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

with col2:
    st.subheader("Diagnostics")
    if st.session_state["history"]:
        st.write("Last user message:", st.session_state["history"][-2][1] if len(st.session_state["history"])>=2 else "-")
    if st.button("Show NLU for last message"):
        if st.session_state["history"]:
            last_user = st.session_state["history"][-2][1]
            nlu = predict_intent(model, le, last_user)
            ents = extract_entities(last_user)
            st.json({"intent": nlu, "entities": ents})
        else:
            st.info("No messages yet")

    st.markdown("---")
    st.subheader("Session State")
    dm = st.session_state["dm"]
    session = dm.get_session(st.session_state["uid"])
    st.write("state:", session.state)
    st.write("slots:", dict(session.slots))
    if st.button("Show full history"):
        st.json(session.history)


# footer
st.markdown("---")
st.caption("Tip: expand intents_spec at the top of the file to add more intents, examples, and slots.")
