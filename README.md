# Email Spam Detection with Word2Vec and SVM

This project aims to build a machine learning model to classify emails into spam and non-spam categories using Word2Vec embeddings and a Support Vector Machine (SVM) classifier.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Dependencies](#dependencies)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)


## Introduction
Spam detection is a common problem in email services. This project uses Word2Vec embeddings to represent email content and an SVM classifier to distinguish between spam and non-spam emails. The dataset contains labeled email messages.

## Data
The dataset should be a CSV file named `Spam_Email_Data.csv` with at least two columns:
- `text`: The content of the email.
- `target`: The label indicating whether the email is "spam" or "non-spam".

## Dependencies
Ensure you have the following libraries installed:
- pandas
- numpy
- nltk
- gensim
- scikit-learn

You can install these dependencies using pip:
```bash
pip install pandas numpy nltk gensim scikit-learn
```

## Preprocessing
The preprocessing steps include:
- Converting text to lowercase
- Removing non-word characters
- Tokenizing the text

## Feature Extraction
The feature extraction process involves:
1. Training a Word2Vec model on the tokenized email content.
2. Generating average Word2Vec vectors for each email.

## Model Training
The model training involves:
- Splitting the dataset into training and testing sets.
- Standardizing the feature vectors.
- Training an SVM classifier on the training set.

## Evaluation
The model is evaluated using metrics such as:
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Usage
1. Clone the repository and navigate to the project directory.
2. Ensure you have the `Spam_Email_Data.csv` file in the directory.
3. open the `Spam_Email_Detection.ipynb` notebook.
4. Run the cells sequentially to execute the data preprocessing, feature extraction, model training, and evaluation steps.

