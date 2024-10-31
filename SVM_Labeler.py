import pandas as pd
from joblib import load

# Load pre-trained model and vectorizer
model_path = 'Joblib/svm_model_a.joblib'  # or 'svm_model_b.joblib'
vectorizer_path = 'Joblib/vectorizer.joblib'
svm_model = load(model_path)
vectorizer = load(vectorizer_path)

# Category mapping
category_mapping = {
    0: 'tech',
    1: 'business',
    2: 'sport',
    3: 'politics',
    4: 'entertainment'
}

# Load the test dataset
bbc_test_path = 'Data/BBC_train_3_tokens_unlabeled.csv'
bbc_test_data = pd.read_csv(bbc_test_path)
X_test_tokens = bbc_test_data['text']

# Transform the test data using the loaded vectorizer
X_test_transformed = vectorizer.transform(X_test_tokens)

# Predict probabilities (if the SVM was trained with `probability=True`) and labels
if hasattr(svm_model, "predict_proba"):
    probabilities = svm_model.predict_proba(X_test_transformed)
    predicted_labels_indices = probabilities.argmax(axis=1)
    predicted_labels = [category_mapping[index] for index in predicted_labels_indices]
    prob_df = pd.DataFrame(probabilities, columns=[f'prob_{category_mapping[i]}' for i in range(len(svm_model.classes_))])
else:
    predicted_labels_indices = svm_model.predict(X_test_transformed)
    predicted_labels = [category_mapping[index] for index in predicted_labels_indices]
    prob_df = pd.DataFrame({'predicted_probability': predicted_labels})

# Add the predicted labels to the test dataset
bbc_test_data['predicted_category'] = predicted_labels

# Combine probabilities (if available) and predicted labels with the original test dataset
if 'prob_df' in locals():
    bbc_test_predictions_with_prob = pd.concat([bbc_test_data, prob_df], axis=1)
else:
    bbc_test_predictions_with_prob = bbc_test_data

# Save the enhanced output to a new CSV file
predictions_output_path = 'Data/BBC_3_SVM_Predictions.csv'
bbc_test_predictions_with_prob.to_csv(predictions_output_path, index=False)

# Print a message indicating the file has been saved
print(f"Predictions have been saved to {predictions_output_path}")
