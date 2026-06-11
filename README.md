# Fake News Detection using NLP & Machine Learning

## Project Overview
The spread of misinformation online is a growing challenge that influences public opinion and undermines trust in media. This project builds an end-to-end Fake News Detection system that classifies news articles as Fake or Real using Natural Language Processing (NLP) and classical Machine Learning techniques.

The project covers the full ML pipeline — from raw data ingestion and text preprocessing through feature engineering, model training, evaluation, and a deployable prediction system.


## Live Demo

Try the deployed Streamlit app here:

[https://your-app-name.streamlit.app](https://fake-news-detection-nlp-project.streamlit.app/)


## Objectives

- Build a binary text classifier to detect fake vs. real news articles
- Apply a complete NLP preprocessing pipeline (normalization → tokenization → stopword removal → stemming → lemmatization)
- Represent text as numerical features using Bag-of-Words and TF-IDF (unigram & bigram)
- Train and compare three ML models: Naive Bayes, Logistic Regression, and SVM
- Identify the most predictive words for fake and real news
- Save the best model for deployment

## Project Structure
```
fake-news-detection-nlp/
│
├── data/
│   ├── true.csv
│   └── fake.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── notebook/
│   └── nlp_fake_news_detection.ipynb
│
├── app.py
└── requirement.txt
└── README.md
```
## Dataset

- Source: [Fake News Detection — Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
- Files: true.csv (real news) and fake.csv (fake news), merged into a single shuffled dataframe
- Total Samples: ~44,000+ articles
- Labels: 1 = Real News, 0 = Fake News
- Columns: title, text, subject, date


## NLP Pipeline

| Step | Description |
|---|---|
| **Normalization** | Lowercase, remove URLs, HTML tags, punctuation, numbers |
| **Tokenization** | Word-level tokenization using NLTK `word_tokenize` |
| **Stopword Removal** | Remove common English stopwords using NLTK |
| **Stemming** | Reduce words to root form using `PorterStemmer` |
| **Lemmatization** | Dictionary-based normalization using `WordNetLemmatizer` |
| **Vectorization** | Count Vectorizer (BoW), TF-IDF (unigram), TF-IDF (bigram) |


## Models & Results

| Model | Vectorization | Accuracy |
|---|---|---|
| Naive Bayes | Count Vectorizer | ~95% |
| Logistic Regression | TF-IDF Unigram | ~98% |
| **SVM (LinearSVC)** | **TF-IDF + Bigram** | **~99.57%** |

> **SVM with TF-IDF Bigram** achieved the best performance. Capturing word context through n-grams significantly improves classification accuracy.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## Requirements

```
pandas
numpy
scikit-learn
nltk
joblib
matplotlib
seaborn
streamlit
```


## Future Improvements

- Integrate transformer-based models (BERT, RoBERTa) for deeper semantic understanding
- Add real-time news scraping for live classification
- Deploy on Streamlit Cloud or Hugging Face Spaces
