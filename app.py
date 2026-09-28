import streamlit as st
import joblib

# Load the complete model pipeline
model = joblib.load("sentiment_model.pkl")

# App title
st.title("Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon review and the model will predict "
    "whether the review is positive or negative."
)

# User input
review = st.text_area("Enter your review:")

# Prediction
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        prediction = model.predict([review])[0]

        if prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")