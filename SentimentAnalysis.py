# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
# Sentiment column: 0 = Negative, 1 = Positive

df = pd.read_csv("C:\\Users\\jyots\\Desktop\\CodTech\\Task-2_SENTIMENT ANALYSIS  WITH NLP\\customer_reviews.csv")

# Display the first few rows of the dataset
df.head()

# Step 3: Preprocessing the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Apply preprocessing to the reviews
df['processed_review'] = df[' Review'].apply(preprocess_text)

# Step 4: Split the data into training and testing sets
X = df['processed_review']
y = df['Liked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Model Analysis
# Visualizing the top features (words) contributing to positive and negative sentiment
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_.flatten()

# Get the top 10 positive and negative words
top_positive_coefficients = coefficients.argsort()[-10:]
top_negative_coefficients = coefficients.argsort()[:10]

top_positive_words = feature_names[top_positive_coefficients]
top_negative_words = feature_names[top_negative_coefficients]

print("Top 10 positive words contributing to positive sentiment:")
print(top_positive_words)

print("\nTop 10 negative words contributing to negative sentiment:")
print(top_negative_words)
