from flask import Flask, render_template, request, send_file
from joblib import load
from Customer_Feedback_Analysis_Tool import preprocess_texts
import pandas as pd
from io import BytesIO

app = Flask(__name__)
model = load("output/model_logreg.joblib")  # Load your trained model
predictions_memory = None  # Store predictions for table + CSV

@app.route("/", methods=["GET", "POST"])
def home():
    global predictions_memory
    results = None
    if request.method == "POST":
        reviews_text = request.form.get("reviews")
        if reviews_text:
            reviews = [r.strip() for r in reviews_text.splitlines() if r.strip()]
            clean_reviews = preprocess_texts(reviews)
            preds = model.predict(clean_reviews)
            predictions_memory = list(zip(reviews, preds))
            results = predictions_memory
    return render_template("index.html", results=results)

@app.route("/download")
def download():
    global predictions_memory
    if not predictions_memory:
        return "No predictions available."
    df = pd.DataFrame(predictions_memory, columns=["Review", "Sentiment"])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
