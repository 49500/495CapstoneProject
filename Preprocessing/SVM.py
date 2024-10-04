import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels):
    # Initialize LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(categories)  # Fit on predefined categories

    # Encode training labels
    encoded_train_labels = encoder.transform(train_labels)

    # Convert training text data to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')  # Change parameters as needed
    X_train = vectorizer.fit_transform(train_texts)

    # Convert test text data to TF-IDF vectors
    X_test = vectorizer.transform(test_texts)

    # Encode test labels
    encoded_test_labels = encoder.transform(test_labels)

    # Train SVM model
    svm_model = SVC(kernel='linear', class_weight='balanced')
    svm_model.fit(X_train, encoded_train_labels)

    # Predict on the test data
    y_pred = svm_model.predict(X_test)

    # Save model, vectorizer, and confusion matrix
    dump(svm_model, 'svm_model.joblib')
    dump(vectorizer, 'vectorizerSVM.joblib')

    # Evaluate the model
    print("Accuracy:", accuracy_score(encoded_test_labels, y_pred))
    print("Classification Report:\n", classification_report(encoded_test_labels, y_pred, target_names=categories, zero_division=0))

    # Save confusion matrix
    cm = confusion_matrix(encoded_test_labels, y_pred)
    np.save('confusion_matrix.npy', cm)

# Replace this part with your actual data for testing
if __name__ == "__main__":
    # Example data - replace with your actual data
    train_texts = ["Sample tech text", "Sample business text", "Sample sport text"]  # Replace with actual texts
    train_labels = ["tech", "business", "sport"]  # Replace with actual labels
    test_texts = ["Sample test tech text", "Sample test business text"]
    test_labels = ["tech", "business"]  # Replace with actual labels

    train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels)
