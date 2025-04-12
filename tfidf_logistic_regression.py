from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from email.parser import BytesParser
from email.policy import default
from bs4 import BeautifulSoup
import numpy as np
import os
import glob
import joblib

# Function to extract email content from .eml files / text from .txt files (AI-generated)
def extract_email_content(file_path):
    if ".txt" in file_path:
        with open(file_path, 'rb') as f:
            text = b" ".join(f.readlines()).decode()
            text = " ".join(text.split())
            return text

    with open(file_path, 'rb') as f:
        email = BytesParser(policy=default).parse(f)
        
        # Extract subject
        subject = email.get('Subject', '')

        # Process multiparts
        text = ""
        if email.is_multipart():  # multipart
            for part in email.iter_parts():
                if part.get_content_type() == 'text/plain':
                    text += part.get_payload(decode=True).decode(errors='ignore')
                elif part.get_content_type() == 'text/html':
                    html = part.get_payload(decode=True).decode(errors='ignore')
                    soup = BeautifulSoup(html, 'html.parser')
                    text += soup.get_text()
        else:  # non-multipart
            text = email.get_payload(decode=True).decode(errors='ignore')
        
        # Remove HTML tag and blanks
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = " ".join(text.split())

        # Add Subject
        full_text = subject+" " +text

        return full_text

# Load and preprocess dataset
def load_dataset(directory):
    emails, labels = [], []
    for label, folder in [('ham', 'ham'), ('spam', 'spam'), ('ai_spam', 'ai_spam')]:
        for file_path in glob.glob(os.path.join(directory, folder, '*')):
            emails.append(extract_email_content(file_path))
            if label == 'ham':
                labels.append(0)  # Normal email
            elif label == 'spam':
                labels.append(1)  # Phishing email (human-gen)
            else:
                labels.append(2)  # Phishing email (AI-gen)
    return emails, labels

# Train and evaluate model
def train_and_evaluate(directory):
    emails, labels = load_dataset(directory)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.8, random_state=42)

    # Config TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.85)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(multi_class='ovr', C=10, max_iter=300)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    # =============================================================#
    # Comment this part if you already got specific hyperparameter
    # =============================================================#

    # # Define Hyperparameter range
    # param_dist = {
    #     'C': np.logspace(-4, 1, 10),  # log scale
    #     'max_iter': [100, 200, 300, 400, 500]
    # }

    # # Execute random search
    # random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=1)
    # random_search.fit(X_train_vec, y_train)

    # # Find best parameters
    # print("Best Parameters:", random_search.best_params_)
    # print("Best Cross-validation Score:", random_search.best_score_)

    # # Predict with best parameters
    # best_model = random_search.best_estimator_
    # predictions = best_model.predict(X_test_vec)

    # ==============================================#

    # Evaluation
    print("Accuracy on Test Set:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Usage
dataset_path = '/home/ubuntu/CS6262/project/dataset'  # Adjust the path as needed
train_and_evaluate(dataset_path)
