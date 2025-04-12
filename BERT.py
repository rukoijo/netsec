import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr

# Step 1. Define the data for spam classification.
# You can expand these lists with more realistic examples for production.
data = {
    "train": {
        "email": [
            "Congratulations! You've won a free cruise. Claim your prize now!",
            "Limited time offer: Buy cheap meds online!",
            "Please review the attached report for your weekly meeting.",
            "Your account statement for last month is ready.",
        ],
        "category": ["spam", "spam", "not spam", "not spam"]
    },
    "test": {
        "email": [
            "Earn money fast with this one weird trick!",
            "Your scheduled meeting has been confirmed."
        ],
        "category": ["spam", "not spam"]
    }
}

# Step 2. Update the label mapping for binary classification.
label_map = {"spam": 0, "not spam": 1}

# Create the training and testing dictionaries with proper labels.
train_data = {
    "email": data["train"]["email"],
    "label": [label_map[cat] for cat in data["train"]["category"]]
}
test_data = {
    "email": data["test"]["email"],
    "label": [label_map[cat] for cat in data["test"]["category"]]
}

# Convert dictionaries into Hugging Face Datasets.
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Group the datasets in a DatasetDict.
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Step 3. Load the tokenizer and the model.
# Make sure to set num_labels=2 for binary classification.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# Step 4. Tokenize the email texts.
def tokenize_function(examples):
    # Tokenizes the 'email' field of the examples.
    return tokenizer(examples["email"], padding="max_length", truncation=True)

# Apply the tokenization to the dataset and remove the original text column.
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["email"])

# Step 5. Define the evaluation metrics.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# Step 6. Set up the training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Create the Trainer instance.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 7. Train the model.
trainer.train()

# Save the trained model and tokenizer.
model.save_pretrained("./email-classifier")
tokenizer.save_pretrained("./email-classifier")

# Step 8. Create the prediction function for the Gradio interface.
def predict_email_category(email):
    # Tokenize the input email text.
    inputs = tokenizer(email, padding=True, truncation=True, return_tensors="pt")
    # Get the model outputs.
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    # Inverse mapping to convert numeric prediction to label.
    inverse_label_map = {v: k for k, v in label_map.items()}
    return inverse_label_map[predicted_class]

# Step 9. Build the Gradio interface.
with gr.Blocks() as demo:
    gr.Markdown("# Email Spam Classifier")
    gr.Markdown("Enter an email text to predict whether it is spam or not spam.")

    email_input = gr.Textbox(label="Email Text")
    output_label = gr.Label(label="Predicted Label")

    submit_button = gr.Button("Classify Email")
    submit_button.click(predict_email_category, inputs=email_input, outputs=output_label)

# Launch the interface.
demo.launch()
