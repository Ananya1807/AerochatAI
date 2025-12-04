AeroChat – Dialogflow-Style Travel Chatbot (Streamlit + NLU)

AeroChat is a lightweight conversational AI assistant built using Python, Streamlit, Scikit-learn, and Regex-based entity extraction.
It simulates Dialogflow-like behavior with intent classification, multi-turn dialogue management, slot filling, and a fully interactive UI.

🚀 Features
🔹 1. Intent Classification (NLU)

Uses TF-IDF + LinearSVC for text classification

Includes sample intents: greet, book_flight, provide_date, goodbye

Model retrains instantly with the built-in “Retrain Model” button

Saves model using joblib

🔹 2. Entity Extraction

Extracts useful details from user messages using regex:

Date (2025-12-20, tomorrow, next Monday, etc.)

Origin (from Mumbai…)

Destination (…to Paris)

Auto-detects capitalized city names

🔹 3. Multi-Turn Dialogue Manager

Implements slot-filling similar to Dialogflow:

Collects origin, destination, date

Tracks session state (idle → collecting → confirm → done)

Stores interaction history

Generates contextual responses

🔹 4. Streamlit Chat UI

Clean two-column layout

Chat window with history

Diagnostics panel showing:

Last user message

Predicted intent & confidence

Extracted entities

Session slots

Full dialogue history

“Reset Conversation” button
