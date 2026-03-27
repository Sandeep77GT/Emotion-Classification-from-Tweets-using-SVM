# Emotion Classification from Tweets using Support Vector Machines

## Overview
This project implements a machine learning pipeline for classifying emotions in tweets using Support Vector Machines (SVM). The system processes raw tweet text, performs preprocessing and feature extraction using TF-IDF, and evaluates multiple SVM kernels to determine the best-performing model.

The project demonstrates a comparative analysis of kernel functions and their impact on text classification performance.

---

## Problem Statement
Understanding emotions expressed in text is a key task in Natural Language Processing with applications in sentiment analysis, social media monitoring, and user behavior analysis. This project aims to:

- Classify tweets into emotion categories
- Evaluate SVM performance with different kernels
- Identify the most effective kernel for emotion classification

---

## Dataset Description

The dataset consists of tweets labeled with emotion categories.

### Features:
- `text`: Raw tweet content
- `label`: Emotion category (e.g., joy, anger, sadness, etc.)

---

## Methodology

### 1. Data Preprocessing
- Removal of URLs, mentions, hashtags, and special characters
- Lowercasing text
- Removal of numbers and punctuation
- Tokenization
- Stopword removal using NLTK
- Stemming using Porter Stemmer

---

### 2. Label Encoding
- Emotion labels converted into numeric format using LabelEncoder

---

### 3. Train-Test Split
- Stratified split to maintain class distribution
- Training set: 80%
- Testing set: 20%

---

### 4. Feature Extraction
- TF-IDF Vectorization
- Maximum features: 20,000
- Converts text data into numerical vectors

---

### 5. Model Training

Support Vector Machine models trained using different kernels:

- Linear
- Polynomial
- Radial Basis Function (RBF)
- Sigmoid

---

### 6. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Experimental Results

- Multiple SVM kernels evaluated and compared
- Performance metrics computed for each kernel
- Best-performing kernel selected based on accuracy and F1-score

### Output Includes:
- Classification reports for each kernel
- Confusion matrices for visual evaluation
- Comparative performance bar charts

---

## Model Persistence

The following components are saved for reuse:

- Best SVM Model (`best_svm_model.pkl`)
- TF-IDF Vectorizer (`tfidf_vectorizer.pkl`)
- Label Encoder (`label_encoder.pkl`)

---

## Project Structure

├── emotions.csv
├── svm_emotion_classifier.py
├── best_svm_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── README.md

---

## Installation

### 1. Clone the Repository
git clone https://github.com/your-username/emotion-classification-svm.git

cd emotion-classification-svm

### 2. Install Dependencies
pip install -r requirements.txt

---

## Running the Project

---

## Running the Project
python svm_emotion_classifier.py

---

## Model Details

- Algorithm: Support Vector Machine (SVM)
- Kernels: Linear, Polynomial, RBF, Sigmoid
- Feature Engineering: TF-IDF
- Text Processing: NLTK-based preprocessing

---

## Key Insights

- Kernel choice significantly impacts classification performance
- Linear and RBF kernels generally perform well for text data
- TF-IDF effectively captures semantic importance of words
- Proper preprocessing improves model accuracy

---

## Limitations

- Computationally expensive for large datasets
- No hyperparameter tuning (C, gamma, degree)
- Limited to traditional ML approach (no deep learning)
- Performance depends on dataset quality and balance

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Use advanced embeddings (Word2Vec, GloVe, BERT)
- Implement deep learning models (LSTM, Transformers)
- Deploy as a web application
- Add real-time inference API

---

## Conclusion

This project demonstrates the effectiveness of Support Vector Machines for emotion classification in text. Through comparative kernel analysis, the study highlights the importance of kernel selection and preprocessing in achieving optimal performance.

---

## Author

Sandeep S L  
MSc Data Analytics (Computational Science)  
Digital University of Kerala  

---

## License

This project is open-source and available under the MIT License.
