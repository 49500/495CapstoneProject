import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Define file paths
true_labels_path = 'Data/BBC_train_3_labels.csv'
svm_predictions_path = 'Data/BBC_3_SVM_Predictions.csv'
nb_predictions_path = 'Data/BBC_train_3_predictions_with_probabilities.csv'

# Load true labels
true_labels_data = pd.read_csv(true_labels_path)
true_labels = true_labels_data['category']

# Load predictions from SVM and Naive Bayes models
svm_predicted_data = pd.read_csv(svm_predictions_path)
nb_predicted_data = pd.read_csv(nb_predictions_path)

# Extract predicted categories
svm_predicted_labels = svm_predicted_data['predicted_category']
nb_predicted_labels = nb_predicted_data['predicted_category']

# Check for length mismatch
if len(true_labels) != len(svm_predicted_labels) or len(true_labels) != len(nb_predicted_labels):
    print("Warning: Lengths of predictions and true labels do not match!")
else:
    # Calculate accuracy and classification reports for each model
    svm_accuracy = accuracy_score(true_labels, svm_predicted_labels)
    nb_accuracy = accuracy_score(true_labels, nb_predicted_labels)
    
    svm_report = classification_report(true_labels, svm_predicted_labels, target_names=['tech', 'business', 'sport', 'politics', 'entertainment'])
    nb_report = classification_report(true_labels, nb_predicted_labels, target_names=['tech', 'business', 'sport', 'politics', 'entertainment'])

    # Determine which model is more accurate
    better_model = "SVM Model" if svm_accuracy > nb_accuracy else "Naive Bayes Model"
    better_accuracy = max(svm_accuracy, nb_accuracy)
    
    # Print comparison results
    print(f"SVM Model Accuracy: {svm_accuracy:.2f}")
    print("SVM Model Classification Report:\n", svm_report)
    print(f"Naive Bayes Model Accuracy: {nb_accuracy:.2f}")
    print("Naive Bayes Model Classification Report:\n", nb_report)
    print(f"The more accurate model is: {better_model} with an accuracy of {better_accuracy:.2f}")

    # Save comparison results to a file
    results_output_path = 'BBC_Model_Comparison_Report.txt'
    with open(results_output_path, 'w') as f:
        f.write(f"SVM Model Accuracy: {svm_accuracy:.2f}\n")
        f.write("SVM Model Classification Report:\n")
        f.write(svm_report + "\n\n")
        f.write(f"Naive Bayes Model Accuracy: {nb_accuracy:.2f}\n")
        f.write("Naive Bayes Model Classification Report:\n")
        f.write(nb_report + "\n\n")
        f.write(f"The more accurate model is: {better_model} with an accuracy of {better_accuracy:.2f}\n")

    print(f"Comparison report has been saved to {results_output_path}")
