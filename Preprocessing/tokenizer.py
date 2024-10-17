import spacy
import csv
from nltk.stem import PorterStemmer

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    # Add any additional preprocessing steps here
    return text.strip().lower()

def tokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            if len(row) == 2:
                # File with 2 columns: [category, text]
                category = row[0]
                text = row[1]
            elif len(row) == 1:
                # File with 1 column: [text only]
                category = None
                text = row[0]

            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Process the text using spaCy
            doc = nlp(preprocessed_text)

            # Filter tokens: retain numbers, remove single letters, stopwords, and punctuation
            filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.like_num or len(token.text) > 1)]
            
            # Apply stemming to the filtered tokens
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

            # Remove empty tokens
            stemmed_tokens = [token for token in stemmed_tokens if token.strip()]

            # Concatenate tokens into a string for cleaner CSV writing
            tokenized_text = ' '.join(stemmed_tokens)
            
            # Write non-empty filtered tokens to the new CSV file
            if tokenized_text:
                if category:  # If category exists, write both category and tokenized text
                    writer.writerow([category, tokenized_text])
                else:  # If no category, write only the tokenized text
                    writer.writerow([tokenized_text])


# tokenize_file('input.csv', 'output.csv')
