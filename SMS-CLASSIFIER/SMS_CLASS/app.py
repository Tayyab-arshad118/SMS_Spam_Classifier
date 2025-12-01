import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# -------------------- NLTK Setup --------------------
# Create a local nltk_data folder inside your app
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add this folder to nltk search paths
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources into that folder
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# -------------------- Text Preprocessing --------------------
def text_trans(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphabetic tokens
    y = [i for i in text if i.isalpha()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    ps = PorterStemmer()
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# -------------------- Load Model & Vectorizer --------------------
BASE_DIR = os.path.dirname(__file__)
tfidf_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# -------------------- Streamlit App --------------------
st.title("📧 Spam SMS/Email Classifier")
st.markdown("Enter the message below to check if it's Ham (Not Spam) or Spam.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
        # Transform and predict
        transformed_text = text_trans(input_sms)
        vector = tfidf.transform([transformed_text])
        result = model.predict(vector)[0]

        # Show result
        if result == 1:
            st.header("⚠️ Spam")
        else:
            st.header("✅ Not Spam")
