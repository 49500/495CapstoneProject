import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Load and vectorize data functions (unchanged from your current implementation)

def load_data(train_file, test_file, test_labels_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file, header=0)
    test_labels_df = pd.read_csv(test_labels_file, header=0)
    return train_df, test_df, test_labels_df

def vectorize_data(train_df, test_df, test_labels_df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['category']
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_labels_df['category']
    return X_train, y_train, X_test, y_test, vectorizer

# Modified Naive Bayes model with mutual learning

def train_and_evaluate_naive_bayes(train_texts, train_labels, test_texts, test_labels, num_epochs=5):
    model_a = MultinomialNB()
    model_b = BernoulliNB()

    subset_size = min(int(0.8 * train_texts.shape[0]), train_texts.shape[0])
    indices = list(range(train_texts.shape[0]))
    random.shuffle(indices)

    subset_indices_a = indices[:subset_size]
    subset_indices_b = indices[:subset_size]

    X_train_a = train_texts[subset_indices_a]
    y_train_a = train_labels[subset_indices_a]
    
    X_train_b = train_texts[subset_indices_b]
    y_train_b = train_labels[subset_indices_b]

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        model_a.fit(X_train_a, y_train_a)
        model_b.fit(X_train_b, y_train_b)

        preds_a = model_a.predict(train_texts)
        preds_b = model_b.predict(train_texts)

        combined_labels_a = [
            pred if pred == true else true for pred, true in zip(preds_b, train_labels)
        ]
        combined_labels_b = [
            pred if pred == true else true for pred, true in zip(preds_a, train_labels)
        ]

        model_a.fit(train_texts, combined_labels_a)
        model_b.fit(train_texts, combined_labels_b)

    y_pred_a = model_a.predict(test_texts)
    y_pred_b = model_b.predict(test_texts)

    accuracy_a = accuracy_score(test_labels, y_pred_a)
    accuracy_b = accuracy_score(test_labels, y_pred_b)
    report_a = classification_report(test_labels, y_pred_a)
    report_b = classification_report(test_labels, y_pred_b)

    dump(model_a, 'naive_bayes_model_a.joblib')
    dump(model_b, 'naive_bayes_model_b.joblib')

    print(f"Model A Accuracy: {accuracy_a}")
    print(f"Model A Classification Report:\n{report_a}")
    print(f"Model B Accuracy: {accuracy_b}")
    print(f"Model B Classification Report:\n{report_b}")