import os, re, argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

negations = {"not": "NOT_", "never": "NOT_", "no": "NOT_"}

def preprocess_texts(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()
        # Replace negations
        for neg in negations:
            text = re.sub(r'\b' + neg + r'\b', negations[neg], text)
        # Remove non-alphabet
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize & remove stopwords + lemmatize
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        cleaned.append(" ".join(words))
    return cleaned


def rating_to_sentiment(r):
    try: r = float(r)
    except: return None
    if r >= 4: return 'positive'
    elif r <= 2: return 'negative'
    else: return 'neutral'



def load_and_prepare(data_path, text_col='review', label_col=None):
    df = pd.read_csv(data_path)
    if label_col and label_col in df.columns:
        lbl_col = label_col
    else:
        for candidate in ['sentiment', 'label', 'rating']:
            if candidate in df.columns:
                lbl_col = candidate
                break
        else:
            raise ValueError("No label column found")
    df = df[[text_col, lbl_col]].dropna(subset=[text_col])
    if df[lbl_col].dtype.kind in 'biufc':
        df['sentiment'] = df[lbl_col].apply(rating_to_sentiment)
    else:
        df['sentiment'] = df[lbl_col].astype(str).str.lower().map({'positive':'positive','pos':'positive',
                                                                   'negative':'negative','neg':'negative',
                                                                   'neutral':'neutral','neu':'neutral'})
    df = df[df['sentiment'].notna()]
    df = df.rename(columns={text_col: 'review'})
    df['clean_review'] = preprocess_texts(df['review'].tolist())
    return df[['review','clean_review','sentiment']]


def build_and_train(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X, y = df['clean_review'].values, df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    pipelines = {
        'logreg': Pipeline([('tfidf', tfidf), ('clf', LogisticRegression(max_iter=200))]),
        'svc': Pipeline([('tfidf', tfidf), ('clf', LinearSVC(max_iter=2000))]),
        'rf': Pipeline([('tfidf', tfidf), ('clf', RandomForestClassifier(n_estimators=200))])
    }
    param_grids = {
        'logreg': {'clf__C':[0.1,1,5]},
        'svc': {'clf__C':[0.1,1]},
        'rf': {'clf__n_estimators':[100,200],'clf__max_depth':[None,30]}
    }

    best_models = {}
    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        grid = GridSearchCV(pipe, param_grids[name], cv=3, n_jobs=-1, scoring='f1_weighted')
        grid.fit(X_train, y_train)
        print(f"Best params {name}: {grid.best_params_}")
        preds = grid.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        print(f"{name} accuracy:{acc:.4f} f1:{f1:.4f}\n")
        best_models[name] = grid.best_estimator_
        joblib.dump(grid.best_estimator_, os.path.join(out_dir,f"model_{name}.joblib"))

    # Save TF-IDF vectorizer from logreg
    joblib.dump(best_models['logreg'].named_steps['tfidf'], os.path.join(out_dir,'tfidf_vectorizer.joblib'))
    return best_models



def inference(texts, model_path, tfidf_path):
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    texts_clean = preprocess_texts(texts)
    X = tfidf.transform(texts_clean)
    preds = model.predict(X)
    return preds



def main():
    parser = argparse.ArgumentParser(description="Customer Feedback Analysis Tool")
    parser.add_argument('--data', required=True)
    parser.add_argument('--text_col', default='review')
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--out_dir', default='output')
    args = parser.parse_args()
    print("Loading data...")
    df = load_and_prepare(args.data, args.text_col, args.label_col)
    print("Class distribution:\n", df['sentiment'].value_counts())
    print("Training models...")
    build_and_train(df, args.out_dir)
    print("All done!")

if __name__=='__main__':
    main()
