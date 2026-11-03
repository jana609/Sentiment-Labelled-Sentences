# Sentiment Analysis Pipeline using Python & Scikit-Learn

An end-to-end natural language processing (NLP) and machine learning pipeline built to classify text sentiments (Positive vs. Negative) using multi-source review datasets (Amazon, IMDb, and Yelp).

---

## 📊 Project Overview
This project processes, cleans, and vectorizes text data to train and evaluate a **Multinomial Naive Bayes** classifier with Laplace smoothing. 

### Key Features:
* **Data Integration:** Combined labelled sentences from Amazon, IMDb, and Yelp datasets (Total: 2,748 samples).
* **Text Preprocessing:** Automated lowercasing, punctuation/special character removal using regular expressions, and handling duplicates/missing values.
* **Feature Extraction:** Converted text data into numerical word-count vectors using `CountVectorizer` with strict separation between training and testing data to prevent data leakage.
* **Classification Model:** Trained a Multinomial Naive Bayes classifier (`alpha=1.0`).

---

## 📈 Model Performance & Evaluation
The model achieved an overall test accuracy of **~79.5%** with a well-balanced distribution across both classes:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (0)** | 0.79 | 0.80 | 0.79 | 271 |
| **Positive (1)** | 0.80 | 0.79 | 0.80 | 276 |

* **Confusion Matrix:**
  ```text
  [[217,  54],
   [ 58, 218]]
