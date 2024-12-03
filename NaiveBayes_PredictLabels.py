import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump
import matplotlib.pyplot as plt

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

# Generate and save the confusion matrix
y_true = bbc_train_1['category']
y_pred = nb_model.predict(X_train_transformed)
cm = confusion_matrix(y_true, y_pred, labels=nb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, values_format='d')
confusion_matrix_path = 'Charts/confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()

# Print a message indicating the confusion matrix has been saved
print(f"Confusion matrix has been saved to {confusion_matrix_path}")
