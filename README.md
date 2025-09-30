# Customer Feedback Analysis Tool
A web-based tool for analyzing customer reviews and predicting sentiment (positive, negative, neutral) using Python, scikit-learn, and Flask.
## Features
- Preprocesses reviews: lowercasing, punctuation removal, stopwords, negation handling, and lemmatization.
- Trains machine learning models: Logistic Regression, SVM, Random Forest.
- Predicts sentiments of new reviews.
- Provides a Flask-based web interface for easy input and prediction.
- Option to download prediction results as CSV.
## Project Structure
feedback_project/
│
├─ Customer_Feedback_Analysis_Tool.py 
├─ app.py
├─ reviews.csv
├─ requirements.txt 
├─ README.md
├─ .gitignore
└─ templates/
└─ index.html 
## How to Use
1. Install dependencies:
```bash
pip install -r requirements.txt

2. Train models:
python Customer_Feedback_Analysis_Tool.py --data reviews.csv --out_dir output

3. Run web app:
python app.py


**Notes**
- Use a larger dataset locally for better accuracy.  
- First time running: download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')



