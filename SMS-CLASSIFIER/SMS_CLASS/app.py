import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')



def text_trans(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]

    for i in text:
        if i.isalpha():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        ps = PorterStemmer()
        y.append(ps.stem(i))

    return " ".join(y)

import os

# get the folder where this script lives
BASE_DIR = os.path.dirname(__file__)

tfidf_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

st.title("📧 Spam SMS/Email Classifier")
st.markdown("Enter the message below to check if it's Ham (Not Spam) or Spam.")
input_sms = st.text_area("Enter the message")
if st.button("Pridect"):
    transformed_text = text_trans(input_sms)
    vector=tfidf.transform([transformed_text])
    result = model.predict(vector)[0]
    if result ==1:
        st.header("Spam")
    else:

        st.header("not Spam")


