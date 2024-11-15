import pandas as pd
from joblib import load

# Define paths
model_path = 'Joblib/svm_model_a.joblib'         # Path to the trained SVM model
vectorizer_path = 'Joblib/vectorizer.joblib'     # Path to the trained vectorizer
bbc_test_path = 'Data/BBC_train_3_tokens_unlabeled.csv'  # Test data file path
predictions_output_path = 'Data/BBC_3_SVM_Predictions.csv'  # Output file path

# Manually define the categories to map predicted labels
categories = ['business','entertainment' ,'politics' ,'sport' , 'tech']

# Load the trained model and vectorizer
svm_model = load(model_path)
vectorizer = load(vectorizer_path)

# Load the test data
bbc_test_data = pd.read_csv(bbc_test_path)
X_test_tokens = bbc_test_data['text']

# Transform the test data using the loaded vectorizer
X_test_transformed = vectorizer.transform(X_test_tokens)

# Predict probabilities
probabilities = svm_model.predict_proba(X_test_transformed)

# Retrieve the numeric class predictions
predicted_labels_indices = probabilities.argmax(axis=1)

# Map indices to actual category names
predicted_labels = [categories[idx] for idx in predicted_labels_indices]

# Add the mapped category predictions to the test data
bbc_test_data['predicted_category'] = predicted_labels

# Create a DataFrame to store probabilities along with predicted labels
prob_df = pd.DataFrame(probabilities, columns=[f'prob_{cls}' for cls in categories])

# Combine the probabilities and labels with the original test dataset
bbc_test_predictions_with_prob = pd.concat([bbc_test_data, prob_df], axis=1)

# Save the enhanced output with probabilities to the specified CSV file
bbc_test_predictions_with_prob.to_csv(predictions_output_path, index=False)

# Print a message indicating the file has been saved
print(f"Corrected predictions with probabilities have been saved to {predictions_output_path}")
