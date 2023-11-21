# app.py
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline with the specified model
sentiment_pipeline = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

# Load the summarization pipeline with the specified model
summarization_pipeline = pipeline(
    "summarization",
    model="nickmuchi/fb-bart-large-finetuned-trade-the-event-finance-summarizer"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        # Use the sentiment analysis pipeline for the input text
        sentiment_result = sentiment_pipeline(text)

        # Use the summarization pipeline for the input text
        summary_result = summarization_pipeline(text, max_length=3000, min_length=75, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)

        # Extract sentiment label and confidence score
        sentiment = sentiment_result[0]['label']
        score = sentiment_result[0]['score']

        # Extract the summary
        summary = summary_result[0]['summary_text']

        return render_template('index.html', text=text, sentiment=sentiment, score=score, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
