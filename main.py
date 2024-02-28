import streamlit as st
import pickle
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

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

# Load pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
# Set page title and favicon
# Set page title and favicon
st.set_page_config(page_title="Email Spam Detector", page_icon=":warning:")

# Center-align the content
st.markdown("""
    <style>
        .center {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Add title and input box to the app
st.markdown('<h1 class="center">Email Spam Message Detector</h1>', unsafe_allow_html=True)
input_sms = st.text_area("Enter the message:")

# Add button to trigger prediction
if st.button('Predict'):
    # Preprocess the input message
    transform_sms = transform_text(input_sms)

    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.error("This message is SPAM!")
    else:
        st.success("This message is NOT SPAM!")
