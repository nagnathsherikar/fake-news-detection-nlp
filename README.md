# Fake News Detection using NLP & Logistic Regression

A machine learning system that classifies news statements as Real or Fake using NLP-based text processing and Logistic Regression.

## Overview
Fake news spreads quickly on social media and can influence public opinion. This project builds an automated classifier using the LIAR dataset, applying TF-IDF feature extraction and Logistic Regression to distinguish real from fake news statements.

## Dataset
Derived from the LIAR Dataset (`train.csv`, `test.csv`, `valid.csv`) — news statements labeled True (Real) or False (Fake).

## Pipeline
1. Text cleaning: lowercasing, punctuation removal
2. Stopword removal and lemmatization (NLTK)
3. TF-IDF vectorization (unigrams + bigrams)
4. Logistic Regression classifier
5. Evaluation via accuracy, F1-score, confusion matrix
6. K-Fold cross-validation for stability

## Results
- Test Accuracy: ~62%
- Test F1 Score: ~0.71
- Average K-Fold F1 Score: ~0.70

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn (TF-IDF, Logistic Regression)
- NLTK (stopwords, lemmatization)

## Limitations
Pattern-based, not fact-based — does not verify claims against real-time sources, and dataset bias can affect predictions.

## Future Scope
Deep learning models (LSTM/BERT), source credibility features, real-time fact-checking API integration.
