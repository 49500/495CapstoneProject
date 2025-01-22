# Capstone Project
Authors: Samuel Theising, JT Wellspring, Nassim Zerrouak, and Zane Smith.

Will be following the footsteps of Prof. Chowdhury and testing a Mutual Learning Algorithm.

### [Final Report](https://49500.github.io/495CapstoneProject/Report.html)

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
This README was updated on 01/22/25
