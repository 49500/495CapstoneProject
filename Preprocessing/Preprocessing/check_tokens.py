import re
import csv

# Regex pattern to check for words under 2 digits, or anything not fitting once its finished.
pattern = r'([a-zA-Z]{2,})|[\d]+|[¬£]\d+'

def check_string(input_string):
    # Outputs if not matching, ignores if it is.
    if not re.match(pattern, input_string):
        print(f"'{input_string}' does not match the pattern")

def check_tokens(input_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        # Iterates through each cell in a row, before moving on to the next row.
        for row in reader:
            for cell in row:  # Check every cell in every row
                check_string(cell)
    print('File done.')

if __name__ == "__main__":
    input_file = r"data/test_data_tokens.csv"
    check_tokens(input_file)