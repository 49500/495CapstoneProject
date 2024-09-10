import csv
import spacy
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

input_file = r"BBC_train_full.csv"
output_file = r"BBC_train_full_tokens_lemmatized.csv"

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remove currency values but keep the numbers
    text = re.sub(r'([¬£€$])(\d+(?:\.\d+)?)', r'\2', text)
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    # Remove apostrophes
    text = text.replace("'", '')
    return text

def is_valid_token(token):
    return (token.like_num or 
            len(token.text) > 1 or 
            (len(token.text) == 1 and token.text.isalpha()) or
            token.text in ['-', '+', '/', '*'])

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        category = row[0]
        text = row[1]
        
        preprocessed_text = preprocess_text(text)
        doc = nlp(preprocessed_text)
        
        filtered_tokens = [token.lemma_ for token in doc 
                           if not token.is_stop and not token.is_punct and is_valid_token(token)]
        
        filtered_tokens = [token for token in filtered_tokens if token.strip()]
        
        if filtered_tokens:
            writer.writerow([category] + filtered_tokens)

print(f"Processed data has been written to {output_file}")