import requests
from bs4 import BeautifulSoup
import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class DarkWebSearch:
    def __init__(self):
        self.categories = {
            "Hacking": "Cyber security, penetration testing, exploits, vulnerabilities, malware",
            "Drugs": "Illicit substances, narcotics, drug trafficking, cannabis, cocaine",
            "Financial Services": "Money laundering, cryptocurrency scams, stolen credit cards",
            "Electronics": "Stolen electronics, counterfeit devices, hacked smartphones",
            "Weapons": "Firearms, illegal weapons trade, ammunition, explosives",
            "Counterfeit Goods": "Fake documents, counterfeit currency, forged IDs",
            "Personal Data": "Stolen identities, credit card information, SSNs",
            "Services": "Hacking for hire, money laundering, document forgery",
        }
    
    def load_models(self):
        """Lazy-load models to optimize memory usage."""
        logging.info("Loading NLP models...")
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.sentence_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        logging.info("Models loaded successfully.")

    def classify_text(self, text):
        """Classify text using NLP or fallback to keywords."""
        if not text.strip():
            return "Unclassified"

        try:
            if not hasattr(self, "classifier"):
                self.load_models()
            
            # Zero-shot classification
            zero_shot_result = self.classifier(text, list(self.categories.keys()))
            if zero_shot_result["scores"][0] > 0.7:
                return zero_shot_result["labels"][0]

            # Semantic similarity
            text_embedding = self.sentence_model.encode(text)
            category_embeddings = {k: self.sentence_model.encode(v) for k, v in self.categories.items()}
            similarities = {
                cat: cosine_similarity([text_embedding], [cat_embed])[0][0]
                for cat, cat_embed in category_embeddings.items()
            }
            most_similar = max(similarities.items(), key=lambda x: x[1])
            return most_similar[0] if most_similar[1] > 0.5 else "Unclassified"

        except Exception as e:
            logging.error(f"NLP classification failed: {e}")
            return "Unclassified"

    def search_ahmia(self, query):
        """Perform the actual dark web search."""
        try:
            url = f"https://ahmia.fi/search/?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("li", class_="result")

            if not results:
                logging.warning("No results found.")
                return []

            processed_results = []
            for result in results:
                title = result.find("h4").get_text(strip=True) if result.find("h4") else "No title"
                link = f"https://ahmia.fi{result.find('a')['href']}" if result.find("a") else "No link"
                desc = result.find("p").get_text(strip=True) if result.find("p") else "No description"
                onion = result.find("cite").get_text(strip=True) if result.find("cite") else "No address"

                category = self.classify_text(f"{title}. {desc}")
                processed_results.append({
                    "title": title,
                    "category": category,
                    "link": link,
                    "description": desc,
                    "onion_address": onion,
                })

            return processed_results

        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

# Initialize search tool
searcher = DarkWebSearch()

@app.route("/api/search", methods=["POST"])
def search():
    """API route for searching."""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        if len(query) > 100:
            return jsonify({"error": "Query too long (max 100 characters)"}), 400

        results = searcher.search_ahmia(query)
        return jsonify({"query": query, "results": results})

    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def home():
    """Root endpoint."""
    return jsonify({
        "message": "Dark Web Search API",
        "endpoints": {
            "/api/search": "POST with {'query': 'your search term'}",
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
