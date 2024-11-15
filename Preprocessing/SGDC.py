from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import os

# Categories list
categories = ['tech', 'business', 'sport', 'politics', 'entertainment']

def train_and_evaluate_sgdc(train_texts, train_labels, test_texts, test_labels, num_epochs=3):
    # Initialize or load label encoder
    label_encoder_file = "Joblib/label_encoder.joblib"
    if not os.path.exists(label_encoder_file):
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)
        dump(label_encoder, label_encoder_file)
    else:
        label_encoder = load(label_encoder_file)

    encoded_train_labels = label_encoder.transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Initialize or load vectorizer
    vectorizer_file = "Joblib/vectorizer.joblib"
    if not os.path.exists(vectorizer_file):
        vectorizer = TfidfVectorizer(max_features=5000)
        train_vectors = vectorizer.fit_transform(train_texts).toarray()
        dump(vectorizer, vectorizer_file)
    else:
        vectorizer = load(vectorizer_file)

    train_vectors = vectorizer.transform(train_texts).toarray()
    test_vectors = vectorizer.transform(test_texts).toarray()

    # Initialize or load SGDClassifier
    sgdc_model_file = "Joblib/sgdc_model.joblib"
    if not os.path.exists(sgdc_model_file):
        sgdc_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3)
    else:
        sgdc_model = load(sgdc_model_file)

    # Incremental training with multiple epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        sgdc_model.partial_fit(train_vectors, encoded_train_labels, classes=list(range(len(categories))))

    # Save the model
    dump(sgdc_model, sgdc_model_file)

    # Evaluate the model
    predictions = sgdc_model.predict(test_vectors)
    print("Model Accuracy:", accuracy_score(encoded_test_labels, predictions))
    print("Model Classification Report:\n", classification_report(encoded_test_labels, predictions, target_names=categories))
