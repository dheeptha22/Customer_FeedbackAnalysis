# Customer Feedback Analysis Tool

A web-based tool for analyzing customer reviews and predicting sentiment (positive, negative, neutral) using Python, scikit-learn, and Flask.

## Features

- Preprocesses reviews: lowercasing, punctuation removal, stopwords, negation handling, and lemmatization.
- Trains machine learning models: Logistic Regression, SVM, Random Forest.
- Predicts sentiments of new reviews.
- Provides a Flask-based web interface for easy input and prediction.
- Option to download prediction results as CSV.

## Project Structure

┌ feedback_project/
│
├─ Customer_Feedback_Analysis_Tool.py ── Training + preprocessing script
├─ app.py ── Flask web app
├─ reviews.csv ── Sample dataset (~50 reviews)
├─ requirements.txt ── Python dependencies
├─ README.md ── Project instructions
├─ .gitignore ── Ignored files/folders
└─ templates/
└─ index.html ── HTML template for Flask                  

