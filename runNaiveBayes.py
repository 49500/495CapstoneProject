from Preprocessing import load_data, vectorize_data
from joblib import dump, load
from Preprocessing.naiveBayes import train_and_evaluate_naive_bayes
# Example usage with the same logic as before
train_file = 'data/BBC_train_1_tokens.csv'
test_file = 'data/test_data_tokens.csv'
test_labels_file = 'data/test_labels.csv'

# Load data
train_df, test_df, test_labels_df = load_data(train_file, test_file, test_labels_file)

# Vectorize data
X_train, y_train, X_test, y_test, vectorizer = vectorize_data(train_df, test_df, test_labels_df)

# Train and evaluate Naive Bayes models
train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test)


