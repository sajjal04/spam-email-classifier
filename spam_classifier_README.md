# Spam Email Classifier

A machine learning project that classifies emails as **spam** or **ham (not spam)** using Natural Language Processing (NLP) techniques.

## Overview

This project compares two ML models — Naive Bayes and Logistic Regression — to classify emails. The best performing model is saved and used for real-time predictions.

## Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 81.25% |
| Logistic Regression | 81.25% |

## Features
- TF-IDF vectorization with bigram support for feature extraction
- Comparison of Naive Bayes vs Logistic Regression classifiers
- Confusion matrix and classification report for model evaluation
- Trained model saved as `.pkl` for reuse
- Real-time prediction on new emails with confidence score

## Tech Stack
Python · scikit-learn · pandas · NumPy · TF-IDF · NLP

## Project Structure
```
spam-email-classifier/
│
├── spam_classifier.py   # Main ML pipeline
├── spam_model.pkl       # Saved trained model
├── vectorizer.pkl       # Saved TF-IDF vectorizer
└── README.md
```

## How to Run

1. Clone the repository
```bash
git clone https://github.com/sajjal04/spam-email-classifier.git
cd spam-email-classifier
```

2. Install dependencies
```bash
pip install scikit-learn pandas numpy
```

3. Run the classifier
```bash
python spam_classifier.py
```

## How It Works

1. **Data** — 80 labeled emails (40 spam, 40 ham)
2. **Preprocessing** — Text cleaned and converted using TF-IDF vectorizer
3. **Training** — Two models trained and compared
4. **Evaluation** — Accuracy, precision, recall, F1-score reported
5. **Prediction** — Model predicts spam/ham with confidence percentage

## Author
**Sajjal Naeem** — github.com/sajjal04
