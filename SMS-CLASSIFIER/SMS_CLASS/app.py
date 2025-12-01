import nltk
import streamlit as st
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Force downloads to the app's writable folder
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Tell NLTK to look in this directory
nltk.data.path.append(nltk_data_dir)

# Function to clean text
def text_trans(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalpha()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Load your model & vectorizer
BASE_DIR = os.path.dirname(__file__)
tfidf_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

# Streamlit app
st.title("📧 Spam SMS/Email Classifier")
st.markdown("Enter the message below to check if it's Ham (Not Spam) or Spam.")
input_sms = st.text_area("Enter the message")
if st.button("Predict"):
    transformed_text = text_trans(input_sms)
    vector = tfidf.transform([transformed_text])
    result = model.predict(vector)[0]
    st.header("Spam" if result == 1 else "Not Spam")
