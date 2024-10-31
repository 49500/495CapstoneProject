from Preprocessing import load_data, vectorize_data, train_naive_bayes, evaluate_model
from joblib import dump, load
from Preprocessing import mutual_learning_with_confidence, evaluate_model

# File paths
train_file_a = 'Data/BBC_train_1_tokens.csv'
train_file_b = 'Data/BBC_train_2_tokens.csv'
test_file = 'Data/test_data_tokens.csv'
test_labels_file = 'Data/test_labels.csv'

# Train Naive Bayes A
print("===== Training Naive Bayes A =====")
train_a_df, test_df, test_labels_df = load_data(train_file_a, test_file, test_labels_file)
X_train_a, y_train_a, X_test_a, y_test_a, vectorizer_a = vectorize_data(train_a_df, test_df, test_labels_df)
model_a = train_naive_bayes(X_train_a, y_train_a)

# Train Naive Bayes B
print("\n===== Training Naive Bayes B =====")
train_b_df, _, _ = load_data(train_file_b, test_file, test_labels_file)
X_train_b, y_train_b, X_test_b, y_test_b, vectorizer_b = vectorize_data(train_b_df, test_df, test_labels_df)
model_b = train_naive_bayes(X_train_b, y_train_b)

# Evaluate both models before mutual learning
print("\n===== Evaluating Naive Bayes A and B Before Mutual Learning =====")
accuracy_a_before, report_a_before = evaluate_model(model_a, X_test_a, y_test_a)
accuracy_b_before, report_b_before = evaluate_model(model_b, X_test_b, y_test_b)
print(f"Naive Bayes A Accuracy (Before): {accuracy_a_before}")
print(f"Classification Report A (Before):\n{report_a_before}")
print(f"Naive Bayes B Accuracy (Before): {accuracy_b_before}")
print(f"Classification Report B (Before):\n{report_b_before}")

# Apply mutual learning with a confidence threshold
print("\n===== Applying Mutual Learning Between Naive Bayes A and B =====")
model_a_after, model_b_after = mutual_learning_with_confidence(model_a, model_b, X_train_a, y_train_a, X_train_b, y_train_b)

# Evaluate both models after mutual learning
print("\n===== Evaluating Naive Bayes A and B After Mutual Learning =====")
accuracy_a_after, report_a_after = evaluate_model(model_a_after, X_test_a, y_test_a)
accuracy_b_after, report_b_after = evaluate_model(model_b_after, X_test_b, y_test_b)
print(f"Naive Bayes A Accuracy (After Mutual Learning): {accuracy_a_after}")
print(f"Classification Report A (After):\n{report_a_after}")
print(f"Naive Bayes B Accuracy (After Mutual Learning): {accuracy_b_after}")
print(f"Classification Report B (After):\n{report_b_after}")

# Save models after mutual learning
dump(model_a_after, 'Joblib/naive_model_a.joblib')
dump(model_b_after, 'Joblib/naive_model_b.joblib')