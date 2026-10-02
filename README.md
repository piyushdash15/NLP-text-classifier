# 🤖 NLP Text Classifier

> **Turn plain text into meaningful predictions with a lightweight machine learning pipeline.**

An end-to-end **Natural Language Processing (NLP)** project that analyzes user-provided text and classifies it into one of three sentiment categories:

**Positive • Neutral • Negative**

The project combines **text preprocessing, TF-IDF feature extraction, machine learning classification, model serialization, and a Streamlit web interface** to demonstrate how an NLP model can move from training code to a practical, user-facing application.

---

## ✨ What This Project Does

Ever wondered how a machine can understand whether a review sounds positive, negative, or neutral?

This project demonstrates the complete process:

```text
Raw Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Prediction + Confidence Scores
   ↓
Interactive Streamlit App
```

You simply enter a sentence, and the application returns the predicted sentiment along with the model's confidence distribution across the three classes.

---

## 🎯 Project Objective

The main objective is to build a practical NLP classification system that demonstrates:

* Text preprocessing and cleaning
* Feature extraction using **TF-IDF**
* Supervised machine learning for text classification
* Model evaluation using standard classification metrics
* Model serialization and reloading
* Real-time inference
* Input validation and error handling
* Interactive web deployment using Streamlit

This project was developed as part of an **AI/ML technical task round** focused on building an end-to-end machine learning application.

---

## 🧠 How It Works

### 1. Text Input

The user enters a review, feedback message, or general piece of text.

Example:

```text
"The product quality is excellent and I really enjoyed using it."
```

### 2. Text Preprocessing

The input is cleaned before being passed to the model.

Typical preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing unnecessary characters
* Normalizing whitespace
* Removing invalid or extremely short inputs

### 3. TF-IDF Vectorization

Machine learning models cannot directly process raw sentences.

TF-IDF converts text into numerical features by measuring how important words and phrases are within the dataset.

The project uses both:

```text
Unigrams → "excellent"

Bigrams → "very good"
```

This allows the model to capture individual words as well as short phrases.

### 4. Classification

The extracted TF-IDF features are passed to a trained classification model.

The primary pipeline is:

```text
TF-IDF
   +
Logistic Regression
```

An additional **Multinomial Naive Bayes** model can be used for model comparison.

### 5. Prediction

The classifier produces:

```text
Predicted Label
+
Probability Distribution
```

Example:

```text
Positive    91%
Neutral      6%
Negative     3%
```

These values represent the model's predicted class probabilities, not a guarantee that the text is objectively positive or negative.

---

## 🗂️ Dataset

The project uses a labeled CSV dataset containing **2,000 text samples**.

### Dataset format

```csv
text,label
"The product is excellent",positive
"The quality is disappointing",negative
"The package arrived on Tuesday",neutral
```

### Class distribution

| Label     |  Samples |
| --------- | -------: |
| Positive  |      667 |
| Negative  |      667 |
| Neutral   |      666 |
| **Total** | **2000** |

The dataset was created specifically for this project and is suitable for demonstrating the complete NLP training and deployment pipeline.

> **Note:** Because the dataset is synthetic, model performance should be interpreted as a demonstration of the ML workflow rather than as evidence of production-level sentiment understanding.

---

## 🛠️ Tech Stack

| Technology       | Purpose                                 |
| ---------------- | --------------------------------------- |
| **Python**       | Core programming language               |
| **Pandas**       | Dataset loading and manipulation        |
| **NumPy**        | Numerical operations                    |
| **Scikit-Learn** | ML models, preprocessing and evaluation |
| **NLTK / Regex** | Text preprocessing                      |
| **Joblib**       | Model serialization                     |
| **Matplotlib**   | Visualization                           |
| **Seaborn**      | EDA and statistical visualization       |
| **Streamlit**    | Interactive web application             |

---

## 🤖 Models

### Primary Model

**TF-IDF + Logistic Regression**

Why Logistic Regression?

* Lightweight and fast
* Works well with sparse text features
* Easy to train
* Supports probability predictions
* Simple to interpret and deploy

### Comparison Model

**TF-IDF + Multinomial Naive Bayes**

This model can be used as a second baseline to compare classification performance and inference characteristics.

---

## 📊 Model Evaluation

The classifier is evaluated using standard classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

For multi-class evaluation, the averaging strategy is explicitly reported so that the metrics are interpreted correctly.

### Performance

> Add the final measured results here after completing model training and evaluation.

Example format:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |        — |         — |      — |        — |
| Naive Bayes         |        — |         — |      — |        — |

---

## 🌐 Interactive Web Application

The trained pipeline is loaded into a **Streamlit** application, allowing users to perform real-time predictions without running the training notebook.

### Application flow

```text
User Input
    ↓
Validation
    ↓
Saved ML Pipeline
    ↓
TF-IDF Transformation
    ↓
Classification
    ↓
Probability Scores
    ↓
Interactive Result
```

The interface is designed to be simple enough for a first-time user while still exposing useful model information.

### Planned / Deployed Demo

**Live Demo:** `YOUR_STREAMLIT_APP_URL`

Replace the placeholder above with your actual deployed URL.

---

## 📁 Project Structure

```text
nlp-text-classifier/
│
├── data/
│   └── nlp_text_classifier_2000.csv
│
├── models/
│   └── sentiment_pipeline.joblib
│
├── notebooks/
│   └── nlp_training.ipynb
│
├── src/
│   └── preprocessing.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nlp-text-classifier.git
cd nlp-text-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Open the training notebook:

```text
notebooks/nlp_training.ipynb
```

Run the cells in order to:

```text
Load Dataset
      ↓
Clean Text
      ↓
Split Dataset
      ↓
TF-IDF
      ↓
Train Models
      ↓
Evaluate
      ↓
Save Pipeline
```

The trained pipeline should be saved inside:

```text
models/
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will then open in your browser.

---

## 🔐 Model Persistence

Instead of saving only the classifier, the project saves the complete preprocessing and prediction pipeline:

```text
TF-IDF Vectorizer
       +
Classifier
       ↓
sentiment_pipeline.joblib
```

This ensures that the same text-processing logic used during training is available during inference.

The saved pipeline is loaded using:

```python
import joblib

model = joblib.load(
    "models/sentiment_pipeline.joblib"
)
```

---

## 🛡️ Input Validation & Error Handling

The application handles common user-input problems such as:

* Empty text
* Extremely short input
* Excessively long input
* Invalid model state
* Prediction failures

Instead of displaying technical errors to the user, the application provides understandable feedback.

---

## 📌 Example Predictions

### Positive

```text
"The product is fantastic and works exactly as expected."
```

**Prediction:** Positive

### Negative

```text
"The product stopped working after two days."
```

**Prediction:** Negative

### Neutral

```text
"The package arrived on Tuesday and included a user manual."
```

**Prediction:** Neutral

---

## 📚 Key Concepts Demonstrated

This project helped implement several important AI/ML concepts in one workflow:

```text
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
TF-IDF
     ↓
Supervised Learning
     ↓
Classification
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Real-Time Inference
     ↓
Web Deployment
```

---

## ⚠️ Limitations

This project is intended primarily as an educational and technical demonstration.

The classifier may struggle with:

* Sarcasm
* Slang
* Ambiguous statements
* Context-dependent sentiment
* Mixed sentiments in a single sentence
* Text very different from the training data

Because the current dataset is synthetic, real-world language diversity may not be fully represented.

---

## 🔮 Future Improvements

Some possible directions for improving the project are:

* Train on a larger real-world sentiment dataset
* Add more diverse domains such as social media and customer support
* Experiment with word embeddings
* Compare additional ML algorithms
* Fine-tune a transformer model such as BERT or DistilBERT
* Add multilingual sentiment classification
* Add batch CSV prediction
* Track model experiments and metrics
* Introduce explainability for individual predictions

---

## 💡 Why This Project Matters

The interesting part of this project is not just the final prediction.

It demonstrates the complete journey of an ML model:

> **from raw data, to features, to learning, to evaluation, and finally to a usable application.**

That same workflow can be extended to spam detection, customer feedback analysis, support-ticket classification, content moderation, and many other NLP problems.

---

## 👨‍💻 Author

**Piyush Kumar Dash**

B.Tech CSE — Artificial Intelligence & Machine Learning

---

## ⭐ Acknowledgement

Built as an AI/ML project with the goal of understanding how a machine learning model can be developed and transformed into a practical end-user application.

---

## 📄 License

This project is intended for educational and portfolio purposes.
