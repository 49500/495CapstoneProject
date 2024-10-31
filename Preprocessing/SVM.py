from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from .check_tokens import check_tokens
import random

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def train_and_evaluate_svm(train_texts, train_labels, test_texts, test_labels, num_epochs=3):
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

    # Initialize two SVM models with different random states
    model_a = SVC(kernel='linear', probability=True, random_state=42)
    model_b = SVC(kernel='linear', probability=True, random_state=24)

    # Use different subsets of the training data for initial training
    subset_size = min(int(0.8 * len(train_texts)), len(train_texts))  # Ensure subset size is not larger than available data

    subset_indices_a = random.sample(range(len(train_texts)), subset_size)
    subset_indices_b = random.sample(range(len(train_texts)), subset_size)
    
    X_train_a = X_train[subset_indices_a]
    y_train_a = encoded_train_labels[subset_indices_a]
    
    X_train_b = X_train[subset_indices_b]
    y_train_b = encoded_train_labels[subset_indices_b]

    # Mutual learning
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Train both models
        model_a.fit(X_train_a, y_train_a)
        model_b.fit(X_train_b, y_train_b)
        
        # Get predictions and probabilities
        preds_a = model_a.predict(X_train)
        probs_a = model_a.predict_proba(X_train)  # Get prediction probabilities
        preds_b = model_b.predict(X_train)
        probs_b = model_b.predict_proba(X_train)  # Get prediction probabilities

        # Update labels based on confidence thresholds
        combined_labels_a = [
            pred if probs_b[i][pred] > 0.7 else true
            for i, (pred, true) in enumerate(zip(preds_b, encoded_train_labels))
        ]
        combined_labels_b = [
            pred if probs_a[i][pred] > 0.7 else true
            for i, (pred, true) in enumerate(zip(preds_a, encoded_train_labels))
        ]
        
        # Re-train models with combined labels
        model_a.fit(X_train, combined_labels_a)
        model_b.fit(X_train, combined_labels_b)

        # Predict on the test data using both models
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)

    # Save models
    dump(model_a, 'Joblib/svm_model_a.joblib')
    dump(model_b, 'Joblib/svm_model_b.joblib')
    dump(vectorizer,'Joblib/vectorizer.joblib')

    # Evaluate the models
    print("Model A Accuracy:", accuracy_score(encoded_test_labels, y_pred_a))
    print("Model A Classification Report:\n", classification_report(encoded_test_labels, y_pred_a, target_names=categories))
    print("Model B Accuracy:", accuracy_score(encoded_test_labels, y_pred_b))
    print("Model B Classification Report:\n", classification_report(encoded_test_labels, y_pred_b, target_names=categories))