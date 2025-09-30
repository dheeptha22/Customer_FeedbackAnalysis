# Customer Feedback Analysis Tool

A simple tool to analyze customer reviews and predict sentiment (positive, negative, neutral) using Python, scikit-learn, and Flask.

## Features

- Preprocesses reviews (lowercase, remove punctuation and stopwords, handle negations, lemmatization)
- Trains models: Logistic Regression, SVM, Random Forest
- Predicts sentiment for new reviews
- Web interface using Flask
- Download prediction results as CSV

## Project Structure

feedback_project/
│
├─ Customer_Feedback_Analysis_Tool.py   # Train + preprocess script
├─ app.py                               # Flask web app
├─ reviews.csv                          # Sample dataset
├─ requirements.txt                     # Python packages
├─ README.md                            # This file
├─ .gitignore                           # Ignore unnecessary files
└─ templates/
    └─ index.html                       # Web page

## How to Use

1. Install dependencies:
pip install -r requirements.txt
2. Train models:
python Customer_Feedback_Analysis_Tool.py --data reviews.csv --out_dir output
3.Run web app:
python app.py


