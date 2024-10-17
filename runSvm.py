import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from Preprocessing.SVM import train_and_evaluate_svm

# File paths
train_file_1 = r"Data/BBC_train_1_tokens.csv"
test_file = r"Data/test_data_tokens.csv"
test_labels_file = r"Data/test_labels.csv"

#Load and process training data
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
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

print(len(train_texts))
subset_size = 500
subset_indices = random.sample(range(len(train_texts)), subset_size)
train_texts = [train_texts[i] for i in subset_indices]
train_labels = [train_labels[i] for i in subset_indices]

# Load and process test data (unlabeled articles)
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

# Call the function with the appropriate arguments
train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels)