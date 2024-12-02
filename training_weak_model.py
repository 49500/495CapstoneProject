import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
def load_data(file_path):
    return pd.read_csv(file_path)

def train_weak_model(weak_model_path, strong_model_path, vectorizer_path, train_file, test_file, test_labels_file):
    # Load the models
    weak_model = load(weak_model_path)
    strong_model = load(strong_model_path)

    # Load the vectorizer
    vectorizer = load(vectorizer_path)

    # Load training data
    train_df = load_data(train_file)

    # Load test data and labels
    test_df = load_data(test_file)
    test_labels_df = load_data(test_labels_file)

    # Assuming 'text' and 'category' columns are present
    X_train = train_df['text']
    y_train = train_df['category']
    
    X_test = test_df['text']
    y_test = test_labels_df['category']

    # Transform the training data using the loaded vectorizer
    X_train_vectorized = vectorizer.transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test)

    # Use the strong model to make predictions
    strong_model_predictions = strong_model.predict(X_train_vectorized)

    # Use predictions from the strong model to create combined labels
    combined_labels = [
        pred if pred == true else true
        for pred, true in zip(strong_model_predictions, y_train)
    ]

    # Retrain the weak model with the combined labels
    weak_model.fit(X_train_vectorized, combined_labels)

    # Evaluate the weak model on the training set
    weak_model_predictions_train = weak_model.predict(X_train_vectorized)
    accuracy_train = accuracy_score(y_train, weak_model_predictions_train)
    report_train = classification_report(y_train, weak_model_predictions_train)

    print(f"Weak Model Accuracy on Training Set after retraining: {accuracy_train}")
    print("Weak Model Classification Report on Training Set:\n", report_train)

    # Evaluate the weak model on the test set
    weak_model_predictions_test = weak_model.predict(X_test_vectorized)
    accuracy_test = accuracy_score(y_test, weak_model_predictions_test)
    report_test = classification_report(y_test, weak_model_predictions_test)

    print(f"Weak Model Accuracy on Test Set: {accuracy_test}")
    print("Weak Model Classification Report on Test Set:\n", report_test)

# Example usage
if __name__ == "__main__":
    weak_model_path = 'Joblib/naive_bayes_model_a.joblib'  # Replace with the path to the weaker model
    strong_model_path = 'Joblib/svm_model_b.joblib'  # Replace with the path to the stronger model
    vectorizer_path = 'Joblib/vectorizer.joblib'  # Path to the vectorizer used for training the SVM
    train_file = 'Data/BBC_train_2_tokens.csv'  # Path to your training data
    test_file = 'Data/test_data.csv'  # Path to your test data
    test_labels_file = 'Data/test_labels.csv'  # Path to your test labels

    train_weak_model(weak_model_path, strong_model_path, vectorizer_path, train_file, test_file, test_labels_file)