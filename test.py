
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
        
    # Write the header row into the output file
    if header: 
      writer.writerow(header)
        
    # Process each row in the input CSV file
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

        # Filter tokens retain numbers, remove single letters, stopwords, and punctuation
        filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.like_num or len(token.text) > 1)  #
        ]
        
        # Remove empty tokens
        filtered_tokens = [token for token in filtered_tokens if token.strip()]

        # Concatenate tokens into a string for cleaner CSV writing
        tokenized_text = ' '.join(filtered_tokens)
        
        # Write non-empty filtered tokens to the new CSV file
        if tokenized_text:
            if category:  # If category exists, write both category and tokenized text
                writer.writerow([category, tokenized_text])
            else:  # If no category, write only the tokenized text
                writer.writerow([tokenized_text])

print(f"Processed data has been written to {output_file}")

