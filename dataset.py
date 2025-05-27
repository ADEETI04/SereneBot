import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import datetime
import random

# Load dataset
df = pd.read_csv("Emotion_classify_Data.csv")
if 'text' not in df.columns or 'emotion' not in df.columns:
    st.error("Dataset must contain 'text' and 'emotion' columns.")
    st.stop()

# Train model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(df['text'], df['emotion'])

# Predefined responses
emotion_responses = {
    'joy': ["That's wonderful! I'm really happy for you."],
    'sadness': ["I'm here for you. You don't have to go through it alone."],
    'anger': ["I understand you're upset. Let's work through this together."],
    'fear': ["You’re not alone. It’s okay to feel scared sometimes."],
    'love': ["Love is a beautiful feeling. Cherish it!"],
    'surprise': ["That sounds surprising! Tell me more."],
    'neutral': ["I'm here to talk if you need anything."],
    'friendly':["Hey there!How are you feeling today?"]
}

# Self-care suggestions
emotion_tips = {
    'joy': "Keep a gratitude journal to cherish these moments.",
    'sadness': "Try writing down your thoughts or taking a walk.",
    'anger': "Take deep breaths or go for a short walk to cool down.",
    'fear': "Practice grounding techniques or guided breathing.",
    'love': "Spend time with loved ones or express your feelings.",
    'surprise': "Reflect on what surprised you and how it made you feel.",
    'neutral': "Try something new or creative today!"
}

# Session state for conversation & emotions
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'emotions' not in st.session_state:
    st.session_state.emotions = []

# UI
st.title("AI Mental Health Chatbot")
st.markdown("Feel free to express how you're feeling.")
st.write("Type a message above and hit Enter to start the conversation.")
user_input = st.text_input("You:")

if user_input:
    emotion = model.predict([user_input])[0]
    response = random.choice(emotion_responses.get(emotion, ["I'm here to listen."]))
    tip = emotion_tips.get(emotion, "")

    # Save conversation
    st.session_state.messages.append((user_input, response, emotion))
    st.session_state.emotions.append((datetime.datetime.now(), emotion))

# Display conversation
if st.session_state.messages:
    st.subheader("Chat History")
    for user, bot, emotion in st.session_state.messages:
        st.markdown(f"*You:* {user}")
        st.markdown(f"*Bot ({emotion.capitalize()}):* {bot}")

# Display emotion tips
if user_input:
    st.markdown("---")
    st.subheader("Wellness Tip")
    st.markdown(f"{tip}")

# Emotion trend graph
if len(st.session_state.emotions) > 1:
    st.markdown("---")
    st.subheader("Your Emotional Trend")
    emotion_df = pd.DataFrame(st.session_state.emotions, columns=['time', 'emotion'])
    emotion_df['time'] = pd.to_datetime(emotion_df['time'])

    fig, ax = plt.subplots()
    emotion_df['emotion'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel("Count")
    ax.set_title("Emotions Detected So Far")
    st.pyplot(fig)