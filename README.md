# Arabic Companies Sentiment Analysis

This project focuses on sentiment analysis of Arabic companies' reviews. The dataset, sourced from Kaggle, includes reviews for 12 companies with a total of 39k reviews. Reviews are rated as -1 for negative, 0 for neutral, and 1 for positive.

## Project Overview

### Exploratory Data Analysis (EDA)

- **Features Analyzed:** Companies, reviews, and ratings.
- **EDA Insights:** Explored the distribution of ratings across companies and the frequency of different sentiments.

### Sentiment Classification Models

#### 1. Data Preprocessing

- Neutral reviews were excluded due to their scarcity, converting the problem into a binary classification of positive and negative reviews.

#### 2. Machine Learning Experiments

- **Best Performing Models:**
  - Multinomial Naive Bayes
  - Logistic Regression
  - Perceptron
  - Stochastic Gradient Descent
  - Voting Classifier (with Perceptron, Stochastic Gradient Descent, and Logistic Regression)
  - Stacking Classifier (with Logistic Regression, Perceptron, Stochastic Gradient Descent as base estimators, and Logistic Regression as the meta model)
- **Feature Extraction:**
  - Bag of Words (BoW) was used, outperforming TFIDF Vectorizer.
- **Hyperparameter Tuning:**
  - Grid Search for Multinomial Naive Bayes.
  - Randomized Search for Logistic Regression.
- **Model Evaluation Metrics:**
  - Precision, Recall, F1 Score, Accuracy, and Area Under Receiver Operating Characteristics (AUC-ROC).
  - AUC-ROC scores ranged from 0.92 to 0.94 for most models, except Perceptron which had the lowest performance.
- **Excluded Models:**
  - Models like Bagging with Decision Tree, SVC, and KNN were excluded due to lower performance.

#### 3. Deep Learning Experiments

- **Best Performing Models:**
  - Long Short-Term Memory (LSTM)
  - Feedforward Neural Network (FFNN)
  - Convolutional Neural Network (CNN)
- **Embedding Layer:**
  - Pre-trained embeddings from FastText for Egyptian Arabic words were used. The best performance was achieved by making the embedding layer trainable.
- **Hugging Face Experiment:**
  - The `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` model was also tested but achieved significantly lower performance compared to other models.

### Performance and Real-time Sentiment Analysis

- **Prediction Speed:** Machine learning models provided predictions very quickly, with an average prediction time of 4 * 10^(-5) seconds.
- **Deep Learning Models:** Achieved results similar to machine learning models.

### Gradio App for Sentiment Analysis

- A Gradio interface was created for real-time sentiment analysis using the stacking model.

## How to Use

1. **Clone the Repository:**

  ```bash
  git clone https://github.com/AdbulrhmanEldeeb/sa_arabic_companies_reviews.git
  cd sa_arabic_companies_reviews/models
  ```
2. **Load the Model in Python:** 
  ```python
  import joblib
  
  vectorizer = joblib.load('count_vectorizer.pkl')
  stacking_model = joblib.load('stacking_model.pkl')
  
  def predict_review_sentiment(review: str) -> str:
      """
      Predicts the sentiment of a given review using a pre-trained stacking model.
  
      Parameters:
      review (str): The review text to be classified.
  
      Returns:
      str: The sentiment classification ('Positive Review' or 'Negative Review').
      """
      transformed_review = vectorizer.transform([review])
      pred = stacking_model.predict(transformed_review)
      return 'Positive Review' if pred == 1 else 'Negative Review'
  
  # Example Usage:
  review = 'الخدمة سيئة جدا ، هذه أخر مرة أشتري هذا المنتج'
  print(predict_review_sentiment(review))
  ```   
### **Pre-trained Embeddings**

#### **Source:** [FastText Egyptian Embeddings](https://www.kaggle.com/datasets/thraxer/fasttext-egyptian-embedding)

### **Results**

- **Model Comparison:** A comparison of metrics for all models can be found in `results/results.csv`.
