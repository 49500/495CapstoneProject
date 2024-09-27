import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(train_file, test_file, test_labels_file):
    # Load preprocessed training data
    train_df = pd.read_csv(train_file)

    # Load test data and labels
    test_df = pd.read_csv(test_file, header=0)
    test_labels_df = pd.read_csv(test_labels_file, header=0)

    return train_df, test_df, test_labels_df

def vectorize_data(train_df, test_df, test_labels_df, max_features=5000):

    # Vectorize both training and test data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the training data
    try:
        X_train = vectorizer.fit_transform(train_df['text'])
        y_train = train_df['category']
    except KeyError as e:
        print(f"Error: {e}. Check if the 'text' and 'category' columns exist in train_df.")
        return None, None, None, None, None

    # Transform the test data (same vectorizer fitted on training data)
    try:
        X_test = vectorizer.transform(test_df['text'])
        y_test = test_labels_df['category']
    except KeyError as e:
        print(f"Error: {e}. Check if the 'text' column exists in test_df and 'category' column exists in test_labels_df.")
        return None, None, None, None, None

    return X_train, y_train, X_test, y_test, vectorizer

def train_naive_bayes(X_train, y_train):
    # Initialize the Naive Bayes model
    model = MultinomialNB()

    # Train the model using the training data
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Generate classification report
    report = classification_report(y_test, predictions)

    return accuracy, report