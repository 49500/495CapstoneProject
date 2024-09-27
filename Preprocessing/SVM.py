import csv
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from .check_tokens import check_tokens

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels):
    # Initialize LabelEncoder -- converts category labels into numeric values
    encoder = LabelEncoder()
    encoder.fit(categories) # Fit on predefined categories

    # Encode training labels using the same encoder as before
    encoded_train_labels = encoder.transform(train_labels)

    # Convert training text data to TF-IDF vectors, including handling stop words
    vectorizer = TfidfVectorizer(stop_words=None)  # Change stop_words as needed
    X_train = vectorizer.fit_transform(train_texts)

    # Convert test text data to TF-IDF vectors using the same vectorizer from training
    X_test = vectorizer.transform(test_texts)

    # Encode test labels
    encoded_test_labels = encoder.transform(test_labels)

    # Train SVM model using the training data
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, encoded_train_labels)

    # Predict on the test data
    y_pred = svm_model.predict(X_test)

    # Save model
    dump(svm_model, 'svm_model.joblib')

    # Evaluate the model
    print("Accuracy:", accuracy_score(encoded_test_labels, y_pred))
    print("Classification Report:\n", classification_report(encoded_test_labels, y_pred, target_names=categories))