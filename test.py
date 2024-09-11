
import csv
import spacy
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

input_file = r"BBC_train_full.csv"
output_file = r"BBC_train_full_tokens.csv"

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remove currency values 
    # text = re.sub(r'£\d+(?:\.\d+)?|€\d+(?:\.\d+)?|\$\d+(?:\.\d+)?', '', text)
    # Replace hyphens with spaces 
    text = text.replace('-', '')
    # Replace apostrophes with spaces 
    text = text.replace("'", '')
    return text

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
    open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    # Create a CSV reader & writer object
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)  # Skip the header row if present
    writer.writerow(header)  # Write the header row into the output file (optional)

    # Iterate through each row in the CSV
    for row in reader:
        # Assume the text to tokenize is in the correct column
        category = row[0]
        text = row[1]

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Process the text using spaCy
        doc = nlp(preprocessed_text)

        # Filter tokens retain numbers, remove single letters, stopwords, and punctuation
        filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.like_num or len(token.text) > 1)  #
        ]
        
        # Remove empty tokens
        filtered_tokens = [token for token in filtered_tokens if token.strip()]

        # Concatenate tokens into a string for cleaner CSV writing
        tokenized_text = ' '.join(filtered_tokens)
        
        # Write non-empty filtered tokens to the new CSV file
        if filtered_tokens:
            writer.writerow([category] + filtered_tokens)

print(f"Processed data has been written to {output_file}")

