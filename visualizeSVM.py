import numpy as np
import matplotlib.pyplot as plt

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def visualize_confusion_matrix():
    # Load the confusion matrix
    cm = np.load('confusion_matrix.npy')

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    # Adding the confusion matrix values to the plot
    threshold = cm.max() / 2.  # Threshold for coloring the text
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_confusion_matrix()
