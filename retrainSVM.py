import csv
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# File paths
new_training_file = r"Data/BBC_train_2_tokens.csv"
test_file = r"Data/test_data_tokens.csv"
test_labels_file = r"Data/test_labels.csv"
svm_model_file = r"svm_model_test.joblib"
vectorizer_file = r"Joblib/vectorizer.joblib"

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

# Load the existing vectorizer
vectorizer = joblib.load(vectorizer_file)

# Transform the data using the existing vectorizer
train_vectors = vectorizer.transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Load the existing SVM model
svm_model = joblib.load(svm_model_file)

# Train the model on the new data
svm_model.fit(train_vectors, train_labels)

# Predict on the test set
predictions = svm_model.predict(test_vectors)

# Encode the labels (if they are strings) to numerical values for evaluation
label_encoder = LabelEncoder()
encoded_test_labels = label_encoder.fit_transform(test_labels)
encoded_predictions = label_encoder.transform(predictions)

# Print the accuracy of the model
print("Model Accuracy:", accuracy_score(encoded_test_labels, encoded_predictions))

# Print the classification report
categories = label_encoder.classes_  # Get the original category names (i.e., class labels)
print("Model Classification Report:\n", classification_report(encoded_test_labels, encoded_predictions, target_names=categories))

# Save the updated SVM model
joblib.dump(svm_model, r"svm_model_test.joblib")
