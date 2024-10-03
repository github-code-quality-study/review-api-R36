import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any, Dict
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def parse_date(self, date_str: str) -> datetime:
        for fmt in ('%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"date_str {date_str} does not match any known format")

    def __call__(self, environ: Dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            filtered_reviews = reviews

            if 'location' in query_params:
                location = query_params['location'][0]
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if 'start_date' in query_params:
                start_date = self.parse_date(query_params['start_date'][0])
                filtered_reviews = [review for review in filtered_reviews if self.parse_date(review['Timestamp']) >= start_date]

            if 'end_date' in query_params:
                end_date = self.parse_date(query_params['end_date'][0])
                filtered_reviews = [review for review in filtered_reviews if self.parse_date(review['Timestamp']) <= end_date]

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(request_body_size)
                params = parse_qs(request_body.decode('utf-8'))

                if "Location" not in params or "ReviewBody" not in params:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Location and ReviewBody are required fields."}).encode("utf-8")]

                if params['Location'][0] not in ['San Diego, California', 'Denver, Colorado']:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location."}).encode("utf-8")]

                review = {
                    'Location': params['Location'][0],
                    'ReviewBody': params['ReviewBody'][0],
                    'ReviewId': str(uuid.uuid4()),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': self.analyze_sentiment(params['ReviewBody'][0])
                }

                reviews.append(review)

                response_body = json.dumps(review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]
            
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
