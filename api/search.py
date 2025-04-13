import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            "Services": "Hacking for hire, money laundering, document forgery"
        }
        
        # Initialize NLP models
        self.classifier = None
        self.sentence_model = None
        self.initialize_models()

    def initialize_models(self):
        """Load optimized NLP models for Vercel"""
        try:
            logging.info("Loading NLP models...")
            # Using smaller models to fit Vercel's limitations
            self.classifier = pipeline("zero-shot-classification", 
                                    model="typeform/distilbert-base-uncased-mnli")
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
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
                category_embeddings = {k: self.sentence_model.encode(v) 
                                    for k, v in self.categories.items()}
                
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
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

# Initialize the search tool
searcher = DarkWebSearch()

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for dark web search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        if len(query) > 100:
            return jsonify({'error': 'Query too long (max 100 characters)'}), 400
        
        results = searcher.search_ahmia(query)
        return jsonify({
            'query': query,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def home():
    """Root endpoint with API documentation"""
    return jsonify({
        'message': 'Dark Web Search API',
        'endpoints': {
            '/api/search': {
                'method': 'POST',
                'description': 'Search the dark web',
                'parameters': {
                    'query': 'Search term (required)'
                },
                'example': {
                    'request': {'query': 'bitcoin'},
                    'response': {
                        'query': 'bitcoin',
                        'count': 5,
                        'results': [{
                            'title': '...',
                            'category': '...',
                            'link': '...',
                            'description': '...',
                            'onion_address': '...'
                        }]
                    }
                }
            }
        }
    })

# Vercel handler
def handler(event, context):
    from flask import Response
    from werkzeug.wrappers import Request
    from werkzeug.datastructures import Headers

    with app.app_context():
        headers = Headers(event.get('headers', {}))
        request = Request({
            'REQUEST_METHOD': event.get('httpMethod', 'GET'),
            'PATH_INFO': event.get('path', '/'),
            'QUERY_STRING': event.get('queryStringParameters', {}),
            'wsgi.input': event.get('body', ''),
            'headers': headers,
        })
        
        response = app.full_dispatch_request(request)
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True)
        }

if __name__ == '__main__':
    app.run(debug=True)
