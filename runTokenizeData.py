import os
from Preprocessing.tokenizer import tokenize_file

# Define input and output file paths
input_train_file = 'Data/BBC_train_full.csv'
output_train_file = 'Data/BBC_train_full_tokens.csv'
input_test_file = 'Data/test_data.csv'
output_test_file = 'Data/test_data_tokens.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
os.makedirs(os.path.dirname(output_test_file), exist_ok=True)

# Tokenize the training data
print(f"Tokenizing training data from {input_train_file} to {output_train_file}...")
tokenize_file(input_train_file, output_train_file)
print("Training data tokenized successfully.")

# Tokenize the test data
print(f"Tokenizing test data from {input_test_file} to {output_test_file}...")
tokenize_file(input_test_file, output_test_file)
print("Test data tokenized successfully.")

