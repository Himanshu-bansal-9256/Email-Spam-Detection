import streamlit as st
import pickle
import string
import nltk

# NLTK downloads (important for deployment)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---------------- TEXT PREPROCESSING ----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------------- LOAD MODEL ----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- UI ----------------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.markdown("<h1 style='text-align: center;'>📧 Email / SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a message to check whether it is Spam or Not</p>", unsafe_allow_html=True)

st.write("")

input_sms = st.text_area("✉️ Enter your message below:", height=150)

st.write("")

if st.button("🔍 Check Message"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # prediction
        result = model.predict(vector_input)[0]

        # probability
        prob = model.predict_proba(vector_input)[0]
        spam_prob = prob[1] * 100
        ham_prob = prob[0] * 100

        st.write("---")

        if result == 1:
            st.markdown(
                f"""
                <div style="background-color:#ffcccc;padding:15px;border-radius:10px;">
                <h2 style="color:#b30000;">🚫 SPAM MESSAGE</h2>
                <p><b>Spam Probability:</b> {spam_prob:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(spam_prob))

        else:
            st.markdown(
                f"""
                <div style="background-color:#ccffcc;padding:15px;border-radius:10px;">
                <h2 style="color:#006600;">✅ NOT SPAM</h2>
                <p><b>Ham Probability:</b> {ham_prob:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(ham_prob))
