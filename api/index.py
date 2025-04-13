from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import requests
from bs4 import BeautifulSoup
import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app and enable CORS
app = Flask(_name_)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})  # Allow Spring Boot requests

# Dark Web Search Class
class DarkWebSearch:
    def _init_(self):
        self.categories = {
            "Hacking": "Cyber security, penetration testing, exploits, vulnerabilities, malware",
            "Drugs": "Illicit substances, narcotics, drug trafficking, cannabis, cocaine",
            "Financial Services": "Money laundering, cryptocurrency scams, stolen credit cards",
            "Electronics": "Stolen electronics, counterfeit devices, hacked smartphones",
            "Weapons": "Firearms, illegal weapons trade, ammunition, explosives",
            "Counterfeit Goods": "Fake documents, counterfeit currency, forged IDs",
            "Personal Data": "Stolen identities, credit card information, SSNs",
            "Services": "Hacking for hire, money laundering, document forgery"
        }
        
        # Initialize NLP models
        self.classifier = None
        self.sentence_model = None
        self.initialize_models()

    def initialize_models(self):
        """Load the NLP models for classification"""
        try:
            logging.info("Loading NLP models...")
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")

    def classify_text(self, text):
        """Classify text using NLP or fallback to keywords"""
        if not text.strip():
            return "Unclassified"
        
        try:
            # Try NLP classification first
            if self.classifier and self.sentence_model:
                # Method 1: Zero-shot classification
                zero_shot_result = self.classifier(text, list(self.categories.keys()))
                if zero_shot_result['scores'][0] > 0.7:
                    return zero_shot_result['labels'][0]
                
                # Method 2: Semantic similarity
                text_embedding = self.sentence_model.encode(text)
                category_embeddings = {k: self.sentence_model.encode(v) for k, v in self.categories.items()}
                
                similarities = {
                    cat: cosine_similarity([text_embedding], [cat_embed])[0][0]
                    for cat, cat_embed in category_embeddings.items()
                }
                most_similar = max(similarities.items(), key=lambda x: x[1])
                if most_similar[1] > 0.5:
                    return most_similar[0]
        except Exception as e:
            logging.error(f"NLP classification failed: {e}")
        
        # Fallback to keyword matching
        return self.classify_with_keywords(text)

    def classify_with_keywords(self, text):
        """Fallback keyword-based classification"""
        keyword_map = {
            "Hacking": ["hack", "cyber", "exploit", "malware", "DDoS"],
            "Drugs": ["drug", "cocaine", "heroin", "meth", "fentanyl"],
            "Financial": ["bitcoin", "bank", "fraud", "laundering", "PayPal"],
            "Electronics": ["iphone", "laptop", "phone", "camera", "drone"],
            "Weapons": ["gun", "rifle", "ammo", "explosive", "grenade"],
            "Counterfeit": ["fake", "replica", "forged", "knockoff"],
            "Personal Data": ["SSN", "passport", "credit card", "dox"],
            "Services": ["for hire", "hitman", "escrow", "smuggling"]
        }
        
        text_lower = text.lower()
        for category, keywords in keyword_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return "Unclassified"

    def search_ahmia(self, query):
        """Perform the actual dark web search"""
        try:
            url = f"https://ahmia.fi/search/?q={query}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("li", class_="result")

            if not results:
                logging.warning("No results found")
                return []

            processed_results = []
            for result in results:
                title = result.find("h4").get_text(strip=True) if result.find("h4") else "No title"
                link = f"https://ahmia.fi{result.find('a')['href']}" if result.find('a') else "No link"
                desc = result.find("p").get_text(strip=True) if result.find("p") else "No description"
                onion = result.find("cite").get_text(strip=True) if result.find("cite") else "No address"

                category = self.classify_text(f"{title}. {desc}")

                processed_results.append({
                    "title": title,
                    "category": category,
                    "link": link,
                    "description": desc,
                    "onion_address": onion
                })

            return processed_results

        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

# Create the Flask route to handle the POST request
@app.route('/api/aisearch', methods=['POST'])
def aisearch():
    data = request.get_json()  # Retrieve data sent by Spring Boot
    query = data.get("query")  # Access query field

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    searcher = DarkWebSearch()
    results = searcher.search_ahmia(query)

    return jsonify({"results": results})  # Send the results back to the Spring Boot app

# Run the app
if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)