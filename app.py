import os
import re
import joblib
import streamlit as st

# LOAD MODEL & VECTORIZER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# TEXT PREPROCESSING
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# PREDICTION FUNCTION
def predict_news(text):
    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    if prediction == 1:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"

# STREAMLIT UI
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.markdown("""
    <style>
        /* Full width button */
        .stButton > button{
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1.1rem !important;
            width: 100% !important;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #1d4ed8;
        }

        /* Top padding on card */
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding-top: 0px !important;
        }
            
         textarea {
            border: 1.5px solid #93c5fd !important;
            border-radius: 8px !important;
        }

    </style>
""", unsafe_allow_html=True)

with st.container(border=True):

    # Title with top padding fix
    st.markdown("""
        <div style='margin: -1rem -1rem 1rem -1rem;'>
            <h1 style='text-align: center; background-color: #2563eb; color: #ffffff;
                       border-radius: 8px 8px 0px 0px; padding: 0.8rem 1rem;
                       margin: 0; font-size: 1.9rem;'>
                Fake News Detection App
            </h1>
        </div>
        <p style='text-align: center; color: #6b7280; margin-top: 0rem; margin-bottom: 1rem;'>
            Enter a news article below and check whether it is REAL or FAKE.
        </p>
    """, unsafe_allow_html=True)

    user_input = st.text_area("Enter News Text:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = predict_news(user_input)
            st.success(result)

    st.markdown("<hr style='margin: 0.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>" \
                     "Built with NLP &amp; Machine Learning"
                "</p>", unsafe_allow_html=True)
