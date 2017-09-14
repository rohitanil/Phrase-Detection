# Phrase-Detection
A Support Vector Machine model which is trained on labelled chats and trying to predict the main phrases of an unknown chat message. Phrase detection uses NLP techniques to clean the training dataset.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
1. Python (used 2.7.13 ) 
2. Pandas
3. NLTK
4. Numpy
5. Sklearn 
6. Pickle

## Installation

1. Copy Haptik.py and Haptik_Eval.py to the local system.
2. Copy training_data.tsv and eval_data.csv. Update the path of training and evaluation dataset in
both the scripts.
3. Run Haptik.py to create the model. The model will be saved as pickle file.
4. Now run the Haptik_Eval.py 
5. Aternatively, copy the final_model.sav to the local system, update its path in the Haptik_Eval.py
and run the script.
6. Output will be saved in final_output.csv

## Built With
1. Atom 
2. IDLE 
3. Jupyter Notebook
