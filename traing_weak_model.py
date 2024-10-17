import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
def load_data(train_file):
    train_df = pd.read_csv(train_file)
    return train_df

def train_weak_model(weak_model_path, strong_model_path, vectorizer_path, train_file):
    # Load the models
    weak_model = load(weak_model_path)
    strong_model = load(strong_model_path)

    # Load the vectorizer
    vectorizer = load(vectorizer_path)

    # Load training data
    train_df = load_data(train_file)

    # Assuming 'text' and 'category' columns are present
    X_train = train_df['text']
    y_train = train_df['category']

    # Transform the training data using the loaded vectorizer
    X_train_vectorized = vectorizer.transform(X_train)

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
    weak_model_predictions = weak_model.predict(X_train_vectorized)
    accuracy = accuracy_score(y_train, weak_model_predictions)
    report = classification_report(y_train, weak_model_predictions)

    print(f"Weak Model Accuracy after retraining: {accuracy}")
    print("Weak Model Classification Report:\n", report)

# Example usage
if __name__ == "__main__":
    weak_model_path = 'naive_bayes_model_a.joblib'  # Replace with the path to the weaker model
    strong_model_path = 'svm_model_a.joblib'  # Replace with the path to the stronger model
    vectorizer_path = 'vectorizer.joblib'  # Path to the vectorizer used for training the SVM
    train_file = 'data/BBC_train_1_tokens.csv'  # Path to your training data

    train_weak_model(weak_model_path, strong_model_path, vectorizer_path, train_file)
