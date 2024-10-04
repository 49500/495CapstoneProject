import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Categories list (adjust as per your dataset)
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def visualize_naive_bayes_confusion_matrix():
    # Load the confusion matrix
    cm = np.load('confusion_matrix_nb.npy')

    # Create a confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Naive Bayes Model')
    plt.show()

if __name__ == "__main__":
    visualize_naive_bayes_confusion_matrix()
