import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load(
        "model/sentiment_pipeline.joblib"
    )

model = load_model()

st.set_page_config(
    page_title="NLP Text Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NLP Text Classifier")
st.write(
    "Enter text and the model will classify it."
)

text = st.text_area(
    "Enter your text:",
    placeholder="Type a product review..."
)

if st.button("Classify"):

    if not text.strip():
        st.warning("Please enter some text.")

    elif len(text.strip()) < 3:
        st.warning("Please enter a longer sentence.")

    else:

        prediction = model.predict([text])[0]

        probabilities = model.predict_proba([text])[0]

        classes = model.classes_

        st.success(
            f"Prediction: {prediction}"
        )

        st.subheader("Confidence Scores")

        for label, prob in zip(
            classes,
            probabilities
        ):
            st.write(
                f"{label}: {prob:.2%}"
            )

            st.progress(float(prob))

if not text.strip():
    st.error("Input cannot be empty.")

if len(text) > 5000:
    st.error("Please keep the input below 5000 characters.")

if len(text) > 5000:
    st.error("Please keep the input below 5000 characters.")

chart_data = pd.DataFrame({
    
    "Probability": probabilities
})

st.bar_chart(
    chart_data.set_index("Class")
)
