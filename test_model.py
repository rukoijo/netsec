import joblib
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load('/home/ubuntu/CS6262/project/logistic_regression_model.pkl')
vectorizer = joblib.load('/home/ubuntu/CS6262/project/tfidf_vectorizer.pkl')

test_dir = '/home/ubuntu/CS6262/project/dataset/test'
emails = []
filenames = []

for file_name in os.listdir(test_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(test_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            emails.append(content)
            filenames.append(file_name)

class_labels = ["Normal", "Human-Phishing", "AI-Phishing"]

def predict_emails(emails):
    X_test = vectorizer.transform(emails)

    # predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[0]  # 각 클래스 확률
    label_index = np.argmax(probs)
    return class_labels[label_index], probs[label_index]
    # results = pd.DataFrame({'filename': filenames, 'prediction': predictions})
    # results.to_csv('test_predictions.csv', index=False)

if __name__ == "__main__":
    results = []
    for email in emails:
        label, confidence = predict_emails([email])
        results.append((label, confidence))
    
    for filename, (label, confidence) in zip(filenames, results):
        print(f"File: {filename}, Prediction: {label}, Confidence: {confidence:.2%}")