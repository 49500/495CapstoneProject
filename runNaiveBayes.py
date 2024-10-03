from Preprocessing import load_data, vectorize_data, train_naive_bayes, evaluate_model
from joblib import dump, load

# File paths
train_file = 'data/BBC_train_1_tokens.csv'
test_file = 'data/test_data_tokens.csv'
test_labels_file = 'data/test_labels.csv'

# Load data
train_df, test_df, test_labels_df = load_data(train_file, test_file, test_labels_file)

# Vectorize data
X_train, y_train, X_test, y_test, vectorizer = vectorize_data(train_df, test_df, test_labels_df)

# Train Naive Bayes model
model = train_naive_bayes(X_train, y_train)

dump(model, 'naive_model.joblib')

# Evaluate model
accuracy, report = evaluate_model(model, X_test, y_test)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")