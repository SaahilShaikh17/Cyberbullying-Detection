from string import punctuation
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib

lemma = WordNetLemmatizer()

model = joblib.load('Logisticregression.pkl')
model_1 = joblib.load('Randomforest.pkl')
model_2 = joblib.load('naive_bayes_classifier.pkl')

# Import Vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Define the preprocessing function
def DataPrep(text):
    # Apply regular expression substitutions to each element in the Series
    text = text.apply(lambda x: re.sub('<.*?>', '', x))
    text = text.apply(lambda x: re.sub(r'http\S+', '', x))
    text = text.apply(lambda x: re.sub(r'@\S+', '', x))
    text = text.apply(lambda x: re.sub(r'#\S+', '', x))
    text = text.apply(lambda x: re.sub(r'\d+', '', x))
    text = text.apply(lambda x: re.sub(r'[^\w\s]', '', x))

    tokens = text.apply(nltk.word_tokenize)

    # Remove punctuation
    punc = list(punctuation)
    words = tokens.apply(lambda x: [w for w in x if w not in punc])

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = words.apply(lambda x: [w.lower() for w in x if not w.lower() in stop_words])

    # Lemmatization
    words = words.apply(lambda x: [lemma.lemmatize(w) for w in x])

    text = words.apply(lambda x: ' '.join(x))

    return text

def main():
    st.title('Classification of Cyberbullying in Tweets')
    st.write('Enter some input data to get a prediction:')
    input_tweet = st.text_area('Input data:')

    if not input_tweet:
        st.warning('Please enter some input data.')
    else:
        test_pred = pd.Series([input_tweet])
        test_pred = test_pred.apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('english'))]))
        test_pred = test_pred.str.replace('\W', ' ', regex=True)
        test_pred = test_pred.str.lower()
        test_pred = DataPrep(test_pred)

        # Transform the input tweet using the vectorizer
        test_data_transformed = vectorizer.transform(test_pred)

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
