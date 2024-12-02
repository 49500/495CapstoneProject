import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# File paths
train_file_1 = r"Data/BBC_train_2_tokens.csv"
test_file = r"Data/test_data_tokens.csv"
test_labels_file = r"Data/test_labels.csv"
sgdc_model_file = r"Joblib/sgdc_model.joblib"
vectorizer_file = r"Joblib/vectorizer.joblib"
label_encoder_file = r"Joblib/label_encoder.joblib"

# Number of epochs
NUM_EPOCHS = 5

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

train_texts_1, train_labels_1 = load_data(train_file_1)

# Combine training data
train_texts = train_texts_1
train_labels = train_labels_1

# Subset the training data
subset_size = 500
subset_indices = random.sample(range(len(train_texts)), subset_size)
train_texts = [train_texts[i] for i in subset_indices]
train_labels = [train_labels[i] for i in subset_indices]

# Load and process test data
def load_test_data(file_path):
    texts = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            text = row[0].strip()  # Ensure the text is non-empty
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
    vectorizer = TfidfVectorizer(max_features=10516)
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

# Train model over multiple epochs
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    sgdc_model.partial_fit(train_vectors, encoded_train_labels, classes=list(range(len(label_encoder.classes_))))

    # Evaluate model on the test set after each epoch
    predictions = sgdc_model.predict(test_vectors)
    print(f"Epoch {epoch + 1} Accuracy: {accuracy_score(encoded_test_labels, predictions):.4f}")
    print(f"Epoch {epoch + 1} Classification Report:\n", classification_report(encoded_test_labels, predictions, target_names=label_encoder.classes_))

# Save the updated model
joblib.dump(sgdc_model, sgdc_model_file)
