import csv
import joblib
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os
<<<<<<< Updated upstream
=======
import seaborn as sns
import numpy as np
import pandas as pd
>>>>>>> Stashed changes

# File paths
new_training_file = r"Data/BBC_train_2_tokens.csv"
test_file = r"Data/test_data_tokens.csv"
test_labels_file = r"Data/test_labels.csv"
sgdc_model_file = r"Joblib/sgdc_model.joblib"
vectorizer_file = r"Joblib/vectorizer.joblib"
label_encoder_file = r"Joblib/label_encoder.joblib"
charts_folder = r"Charts"

# Number of epochs for training

epochs = 40

# Load and process training data
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            if len(row) == 2:
                category = row[0].strip()
                text = row[1].strip()
                if text:
                    labels.append(category)
                    texts.append(text)
    return texts, labels

# Load new training data
train_texts, train_labels = load_data(new_training_file)

# Load and process test data
def load_test_data(file_path):
    texts = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            text = row[0].strip()
            if text:
                texts.append(text)
    return texts

test_texts = load_test_data(test_file)

# Load test labels
def load_test_labels(file_path):
    labels = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            label = row[0].strip()
            if label:
                labels.append(label)
    return labels

test_labels = load_test_labels(test_labels_file)

# Initialize or load vectorizer
if not os.path.exists(vectorizer_file):
    vectorizer = TfidfVectorizer(max_features=5000)
    train_vectors = vectorizer.fit_transform(train_texts).toarray()
    joblib.dump(vectorizer, vectorizer_file)
else:
    vectorizer = joblib.load(vectorizer_file)

train_vectors = vectorizer.transform(train_texts).toarray()
test_vectors = vectorizer.transform(test_texts).toarray()

# Initialize or load label encoder
if not os.path.exists(label_encoder_file):
    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    joblib.dump(label_encoder, label_encoder_file)
else:
    label_encoder = joblib.load(label_encoder_file)

encoded_train_labels = label_encoder.transform(train_labels)
encoded_test_labels = label_encoder.transform(test_labels)

# Initialize or load SGDClassifier
if not os.path.exists(sgdc_model_file):
    sgdc_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3)
else:
    sgdc_model = joblib.load(sgdc_model_file)

os.makedirs(charts_folder, exist_ok=True)
accuracies = []
# Incremental training with epochs
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    sgdc_model.partial_fit(train_vectors, encoded_train_labels, classes=list(range(len(label_encoder.classes_))))
    predictions = sgdc_model.predict(test_vectors)
    
    accuracy = accuracy_score(encoded_test_labels, predictions)
    print(f"Epoch {epoch + 1} Accuracy: {accuracy:.4f}")
    accuracies.append(accuracy)
    # Predict on the test set
    predictions = sgdc_model.predict(test_vectors)
    cm = confusion_matrix(encoded_test_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
    plt.savefig(os.path.join(charts_folder, f'confusion_matrix_epoch_{epoch + 1}.png'))
    plt.close()
    if accuracy_score(encoded_test_labels, predictions) > 0.97:
        break



# Evaluate model
print("Model Accuracy:", accuracy_score(encoded_test_labels, predictions))
print("Model Classification Report:\n", classification_report(encoded_test_labels, predictions, target_names=label_encoder.classes_))

# Save the updated model
joblib.dump(sgdc_model, sgdc_model_file)
<<<<<<< Updated upstream
=======
import matplotlib.pyplot as plt

# Create Charts folder if it doesn't exist
charts_folder = "Charts"
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)




plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(charts_folder, 'accuracy_over_epochs.png'))
plt.show()

# Plot classification report as a heatmap

report = classification_report(encoded_test_labels, predictions, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(12, 8))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report Heatmap')
plt.savefig(os.path.join(charts_folder, 'classification_report_heatmap.png'))
plt.show()
>>>>>>> Stashed changes
