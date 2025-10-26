from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import torch
import sqlite3
import requests
from datetime import datetime
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os, glob
import feedparser
import urllib.parse

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load models
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Semantic similarity model (SBERT)
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# KeyBERT model
kw_model = KeyBERT()

# Image model
resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

known_embeddings = []

def load_reference_embeddings():
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    for file_path in glob.glob("embeddings/*.npy"):
        name = os.path.basename(file_path).replace(".npy", "")
        embedding = np.load(file_path)
        known_embeddings.append({"name": name, "embedding": embedding})

load_reference_embeddings()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        text = request.json['text']
        if not text.strip():
            return jsonify({'error': 'Empty text input'}), 400

        lower_text = text.lower().strip()

        # Rule-based answers
        if lower_text in ["who created you", "who made you", "who developed you"]:
            return jsonify({
                'label': "INFORMATION",
                'answer': "MUKKA SRIVATSAV",
                'confidence': "100%",
                'sources': ["Self-Identity"]
            })

        elif "did mukka srivatsav made you" in lower_text or "did mukka srivatsav create you" in lower_text:
            return jsonify({
                'label': "REAL",
                'confidence': "100%",
                'sources': ["Self-Identity"]
            })

        sources, relevant = verify_text_google_semantic(text)

        if relevant:
            return jsonify({
                'label': "REAL",
                'confidence': "Verified by Google News",
                'sources': sources
            })

        # Model fallback
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        FAKE_PROB = probs[0][0].item()
        REAL_PROB = probs[0][1].item()

        threshold = 0.6
        if FAKE_PROB > threshold:
            label = "FAKE"
            confidence = FAKE_PROB * 100
        elif REAL_PROB > threshold:
            label = "REAL"
            confidence = REAL_PROB * 100
        else:
            label = "UNCERTAIN"
            confidence = max(FAKE_PROB, REAL_PROB) * 100

        log_analysis(text, label)

        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'sources': sources
        })

    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image = Image.open(file.stream).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        embedding = resnet(tensor).detach().numpy()

        found_name, distance = compare_with_known_embeddings(embedding)

        if found_name:
            label = f"Hello {found_name.replace('_', ' ')}"
        else:
            label = "Unknown Person"

        return jsonify({
            'label': label,
            'distance': float(distance),
            'metadata': {}
        })

    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

# FULL SMART GOOGLE NEWS SEMANTIC VERIFICATION
def verify_text_google_semantic(text):
    try:
        keywords = extract_keywords(text)
        query = urllib.parse.quote(keywords)
        url = f"https://news.google.com/rss/search?q={query}"
        feed = feedparser.parse(url)
        sources = []
        relevant = False

        query_embed = sbert.encode(text, convert_to_tensor=True)

        for entry in feed.entries:
            sources.append(entry.title)
            title_embed = sbert.encode(entry.title, convert_to_tensor=True)
            similarity = util.cos_sim(query_embed, title_embed).item()
            if similarity >= 0.75:  # highly accurate threshold
                relevant = True

        return sources, relevant

    except Exception as e:
        print(f"Google News error: {str(e)}")
        return [], False

# Keyword Extraction
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)
    keyword_string = " ".join([kw[0] for kw in keywords])
    return keyword_string

def compare_with_known_embeddings(embedding, threshold=0.7):
    min_distance = float('inf')
    recognized_name = None

    for entry in known_embeddings:
        dist = np.linalg.norm(embedding - entry["embedding"])
        if dist < min_distance:
            min_distance = dist
            recognized_name = entry["name"]

    if min_distance <= threshold:
        return recognized_name, min_distance
    else:
        return None, min_distance

def log_analysis(input_data, result):
    try:
        conn = sqlite3.connect('logs.db')
        conn.execute('INSERT INTO logs (input, result, timestamp) VALUES (?, ?, ?)',
                     (input_data, result, datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
