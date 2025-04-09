import joblib
import os
import pandas as pd
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

X_test = vectorizer.transform(emails)

predictions = model.predict(X_test)

results = pd.DataFrame({'filename': filenames, 'prediction': predictions})
results.to_csv('test_predictions.csv', index=False)