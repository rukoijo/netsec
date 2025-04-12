import os
import glob
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Set paths to your email directories (adjust the path as needed)
ham_dir = './email-dataset/dataset/1'   # directory for ham emails
spam_dir = './email-dataset/dataset/2'  # directory for spam emails

def load_emails(directory, label):
    """
    Load all .eml files from a directory, clean the text,
    and assign the provided label.
    """
    emails = []
    labels = []
    # Using glob to get all .eml files in the directory
    for filepath in glob.glob(os.path.join(directory, '*.eml')):
        with open(filepath, 'r', encoding='latin1') as f:
            text = f.read()
            # Basic cleaning: remove HTML tags and lower-case the text.
            text = re.sub(r'<[^>]+>', '', text)
            text = text.lower()
            emails.append(text)
            labels.append(label)
    return emails, labels

# Load emails and labels for both ham and spam
ham_emails, ham_labels = load_emails(ham_dir, 0)   # label 0 for ham
spam_emails, spam_labels = load_emails(spam_dir, 1)  # label 1 for spam

# Combine ham and spam data
emails = ham_emails + spam_emails
labels = ham_labels + spam_labels

# Parameters for tokenization and padding
max_words = 10000    # maximum number of words to consider in the vocabulary
max_len = 500        # maximum sequence length per email (adjust based on your data)

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to a NumPy array
y = np.array(labels)

# Split data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a custom CNN model for text classification
embedding_dim = 100  # size of the word vectors

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: spam vs. ham
    #Dense(1, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# step 8: plots the results for thesis (accuracy,loss, validation accuracy, validation loss)
# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('Model loss') # "Learning rate =" + str(0.0001)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
