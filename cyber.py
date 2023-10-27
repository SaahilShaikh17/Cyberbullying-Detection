from string import punctuation
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib

# Download nltk data (add this to your code)
nltk.download('punkt')
nltk.download('stopwords')

lemma = WordNetLemmatizer()

model = joblib.load('Logisticregression.pkl')
model_1 = joblib.load('Randomforest.pkl')
model_2 = joblib.load('naive_bayes_classifier.pkl')

# Import Vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Define the preprocessing function
def DataPrep(text):
    # Apply regular expression substitutions to each element in the Series
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = nltk.word_tokenize(text)

    # Remove punctuation
    punc = list(punctuation)
    words = [w for w in tokens if w not in punc]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in words if not w.lower() in stop_words]

    # Lemmatization
    words = [lemma.lemmatize(w) for w in words]

    text = ' '.join(words)

    return text

def main():
    st.title('Classification of Cyberbullying in Tweets')
    st.write('Enter some input data to get a prediction:')
    input_tweet = st.text_area('Input data:')

    if not input_tweet:
        st.warning('Please enter some input data.')
    else:
        # Preprocess the input data
        input_tweet = DataPrep(input_tweet)

        # Transform the input tweet using the vectorizer
        test_data_transformed = vectorizer.transform([input_tweet])

        # Create a dictionary to store the class labels and probabilities
        class_labels = {
            0: "Age",
            1: "Ethnicity",
            2: "Gender",
            3: "Religion",
            4: "Not_Cyberbullying",
        }

        # Make predictions using all three models
        pred_proba = model.predict_proba(test_data_transformed)[0]
        pred_label = model.predict(test_data_transformed)[0]

        pred_proba_1 = model_1.predict_proba(test_data_transformed)[0]
        pred_label_1 = model_1.predict(test_data_transformed)[0]

        pred_proba_2 = model_2.predict_proba(test_data_transformed)[0]
        pred_label_2 = model_2.predict(test_data_transformed)[0]

        # Display the predictions
        st.header("Predictions:")
        
        st.subheader("Prediction using Logistic Regression:")
        st.write(f"Classification: {class_labels[pred_label]}")
        st.write(f"Probability: {pred_proba[pred_label]:.3f}")

        st.subheader("Prediction using Random Forest:")
        st.write(f"Classification: {class_labels[pred_label_1]}")
        st.write(f"Probability: {pred_proba_1[pred_label_1]:.3f}")

        st.subheader("Prediction using Naive Bayes:")
        st.write(f"Classification: {class_labels[pred_label_2]}")
        st.write(f"Probability: {pred_proba_2[pred_label_2]:.3f}")

if __name__ == '__main__':
    main()
