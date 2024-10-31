import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump

# Load BBC_train_1_tokens and BBC_train_3_tokens_unlabeled
bbc_train_1_path = 'Data/BBC_train_1_tokens.csv'
bbc_train_3_unlabeled_path = 'Data/BBC_train_3_tokens_unlabeled.csv'

bbc_train_1 = pd.read_csv(bbc_train_1_path)
bbc_train_3_unlabeled = pd.read_csv(bbc_train_3_unlabeled_path)

# Prepare the data
X_train_tokens = bbc_train_1['text']
y_train_labels = bbc_train_1['category']
X_test_tokens = bbc_train_3_unlabeled['text'] 

# Initialize and transform text data using CountVectorizer 
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train_tokens)
X_test_transformed = vectorizer.transform(X_test_tokens)

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_transformed, y_train_labels)

# Get the probability for each class using the trained model
probabilities = nb_model.predict_proba(X_test_transformed)

# Retrieve the predicted class labels based on highest probability
predicted_labels = nb_model.classes_[probabilities.argmax(axis=1)]

# Create a DataFrame to store probabilities along with predicted labels
# Get class labels and attach each probability column with its respective class
prob_df = pd.DataFrame(probabilities, columns=[f'prob_{cls}' for cls in nb_model.classes_])
bbc_train_3_unlabeled['predicted_category'] = predicted_labels

# Combine the probabilities and labels with the original test dataset
bbc_train_3_predictions_with_prob = pd.concat([bbc_train_3_unlabeled, prob_df], axis=1)

# Save the trained model and predictions to files
model_path = 'trained_naive_bayes_model.joblib'
dump(nb_model, model_path)

# Save this enhanced output to a new CSV file
predictions_with_prob_output_path = 'Data/BBC_train_3_predictions_with_probabilities.csv'
bbc_train_3_predictions_with_prob.to_csv(predictions_with_prob_output_path, index=False)

# Print a message indicating the file has been saved
print(f"Predictions with probabilities have been saved to {predictions_with_prob_output_path}")
