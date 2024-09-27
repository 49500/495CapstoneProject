# Capstone Project
Authors: Samuel Theising, JT Wellspring, and Nassim Zerrouak.

Will be following the footsteps of Prof. Chowdhury and testing a Mutual Learning Algorithm.
### How to run the program
Must download python. We are using 3.12.5 to run and implement our own version of the project. To view:
```python
python --version
```
We used a built in library to tokenize the data. spaCy install:
```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
As well as the use of Pandas:
```bash
pip install pandas
```
Download all Dependencies
```bash
pip install -e .
```
To Run
```bash
python runTokenizeData.py

python runNaiveBayes.py

python runSvm.py
```
### Last Updated
This README was updated on 9/7/24
