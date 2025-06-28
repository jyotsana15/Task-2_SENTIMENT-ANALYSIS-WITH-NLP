# SENTIMENT ANALYSIS WITH NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JYOTSANA BHARDWAJ

*INTERN ID*: CT08DK599

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

Sentiment Analysis on Customer Reviews using Logistic Regression
📖 Introduction
Introduction  
This project focuses on creating a Sentiment Analysis model to classify customer reviews as either positive or negative. Sentiment Analysis is a key application of Natural Language Processing (NLP) that helps businesses understand customer opinions from text. The model uses Logistic Regression, a common algorithm for binary classification tasks.  

Through this project, I gained hands-on experience in text preprocessing, feature extraction with TF-IDF vectorization, model training, evaluation, and result visualization. 

🔍 Objective
The goal of this project is to:  

- Preprocess raw text reviews to prepare them for machine learning  
- Extract meaningful features from text using TF-IDF  
- Train a Logistic Regression model to classify sentiment  
- Evaluate the model’s performance with metrics like accuracy, a classification report, and a confusion matrix  
- Identify key words that contribute to positive and negative sentiments

🧰 Technologies and Libraries Used
- Python 3  
- Pandas and NumPy for handling and manipulating data  
- re and string libraries for text preprocessing  
- scikit-learn for building the model, TF-IDF vectorization, and evaluation  
- Matplotlib and Seaborn for data visualization

🛠 Dataset Details
The dataset contains customer reviews with corresponding sentiment labels:  
- The Liked column indicates sentiment (0 = Negative, 1 = Positive)  
- The Review column holds raw customer text reviews

🔧 Project Workflow
- Import Libraries: Loaded essential Python libraries for data processing, machine learning, and visualization.  
- Load Dataset: Imported the CSV file with customer reviews and their sentiment labels.  
- Text Preprocessing:  
  - Converted text to lowercase for consistency.  
  - Removed punctuation and numbers using Python’s string and regex functions.  
  - Eliminated extra white spaces for cleaner data.  
- Data Splitting: Split the dataset into training (70%) and testing (30%) sets using train_test_split to fairly evaluate the model.  
- TF-IDF Vectorization:  
  - Transformed text data into numerical features using TfidfVectorizer.  
  - Limited vocabulary to the top 5000 features and removed English stop words to reduce noise.  
- Model Training:  
  - Trained a Logistic Regression classifier on the TF-IDF transformed training data.  
  - Increased maximum iterations to 1000 for better convergence.  
- Prediction and Evaluation:  
  - Predicted sentiments on the test set.  
  - Calculated accuracy to measure overall performance.  
  - Printed the classification report for precision, recall, and F1-score.  
  - Generated and visualized the confusion matrix to analyze true/false positives and negatives.  
- Feature Analysis:  
  - Extracted the most influential words for positive and negative sentiment based on model coefficients.  
  - Displayed the top 10 positive and negative words to interpret the model. 

📊 Results
The model achieved a high accuracy score, typically above 85% depending on the dataset.  
The confusion matrix visualization helped reveal the model’s strengths and weaknesses.  
Top positive words like "great," "love," and "best," along with negative words like "bad," "worst," and "disappointed," provided insights into what drives customer sentiment.

![Image](https://github.com/user-attachments/assets/564fa885-d1df-4e55-8c56-83d7180392eb)

🚀 How to Run
Simply, run the script:
<pre><code>python sentiment_analysis.py</code></pre>
Make sure to place the dataset CSV file in the correct path or update the path in the script accordingly.
