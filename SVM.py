import csv
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# File paths
train_file = r"BBC_train_full_tokens.csv"
test_file = r"test_data_tokens.csv"
test_labels_file = r"test_labels.csv"

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

# Initialize LabelEncoder -- converts category labels into numeric values
encoder = LabelEncoder()
encoder.fit(categories) # Fit on predefined categories

# Step 1: Load and process training data
train_texts = []
train_labels = []

with open(train_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present

    for row in reader:
        if len(row) == 2:  # [category, text]
            category = row[0].strip()
            text = row[1].strip()

            # Ensure text is non-empty
            if text:
                train_labels.append(category)
                train_texts.append(text)

# Encode training labels using the same encoder as before
encoded_train_labels = encoder.transform(train_labels)

# Convert training text data to TF-IDF vectors, including handling stop words
vectorizer = TfidfVectorizer(stop_words=None)  # Change stop_words as needed
X_train = vectorizer.fit_transform(train_texts)

# Step 2: Load and process test data (unlabeled articles)
test_texts = []
with open(test_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present

    for row in reader:
        text = row[0].strip()  # Ensure the text is non-empty
        if text:
            test_texts.append(text)

# Convert test text data to TF-IDF vectors using the same vectorizer from training
X_test = vectorizer.transform(test_texts)

# Step 3: Load the true test labels from test_labels.csv
test_labels = []
with open(test_labels_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present

    for row in reader:
        label = row[0].strip()  # Ensure the label is non-empty
        if label:
            test_labels.append(label)

# Encode test labels
encoded_test_labels = encoder.transform(test_labels)

# Step 4: Train SVM model using the training data
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, encoded_train_labels)

# Step 5: Predict on the test data
y_pred = svm_model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(encoded_test_labels, y_pred))
print("Classification Report:\n", classification_report(encoded_test_labels, y_pred, target_names=categories))
