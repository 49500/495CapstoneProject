import csv
from Preprocessing.SVM import train_and_evaluate_svm
from Preprocessing.check_tokens import check_tokens

# File paths
train_file = r"Data/BBC_train_2_tokens.csv"
test_file = r"Data/test_data_tokens.csv"
test_labels_file = r"Data/test_labels.csv"

# Check tokens in the files
#check_tokens(train_file)
#check_tokens(test_file)
#check_tokens(test_labels_file)

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

# Step 2: Load and process test data (unlabeled articles)
test_texts = []
with open(test_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present

    for row in reader:
        text = row[0].strip()  # Ensure the text is non-empty
        if text:
            test_texts.append(text)

# Step 3: Load the true test labels from test_labels.csv
test_labels = []
with open(test_labels_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present

    for row in reader:
        label = row[0].strip()  # Ensure the label is non-empty
        if label:
            test_labels.append(label)

# Call the function to train and evaluate the SVM model
train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels)