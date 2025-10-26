# Fake News & Image Verification System 🧠🔎

An AI-powered Flask web application that detects fake news content and verifies images using **BERT**, **SBERT**, **KeyBERT**, **ResNet50**, and **Google News Semantic Matching**.

---

## 🚀 Features

- Text-based fake news detection (BERT)
- Semantic verification via Google News results (SBERT)
- Keyword extraction using KeyBERT
- Image authenticity verification (ResNet50)
- Automatic logging via SQLite
- PDF report generation
- Interactive web UI built using React + TailwindCSS

---

## 🧩 Project Structure
project/
│
├── app.py # Flask backend with routes and model logic

├── register_image.py # Utility to add known faces for matching

├── init_db.py # Creates logs.db file

├── index.html # Frontend interface

├── embeddings/ # Stores reference image embeddings

└── logs.db 
# Log database (auto-generated)

---

## ⚙️ Installation

### 1. Clone this repository

git clone https://github.com/M-Srivatsav999/Fake-News-Detection-.git

cd fake-news-detector

### 2. Create environment

python -m venv venv

venv/Scripts/activate # Windows

source venv/bin/activate # macOS/Linux

### 3. Install dependencies

pip install -r requirements.txt

---

## 🧠 Models Used

- **Text detection**: `jy46604790/Fake-News-Bert-Detect`
- **Semantic search**: `all-MiniLM-L6-v2`
- **Keywords**: KeyBERT
- **Image embeddings**: ResNet50 (`torchvision.models.resnet50`)

---

## 🧾 Usage

### Initialize database

python init_db.py

### Register a reference image (optional)

python register_image.py

### Start the Flask server

python app.py

Then open http://localhost:5000 in your browser.

---

## 📊 API Endpoints

| Method | Endpoint         | Description |
|--------|------------------|--------------|
| POST   | /analyze-text    | Detects fake/real news |
| POST   | /analyze-image   | Verifies uploaded image |

Example:
{ "text": "Breaking news: ..." }

---

## 💡 Frontend Details

The `index.html` UI is built with:
- **React** + **Babel**  
- **Axios** for calling Flask API  
- **TailwindCSS** for styling  
- **jsPDF** for generating reports  

---

## 🧰 Requirements

See the `requirements.txt` file below for all dependencies.

---

## 👨‍💻 Author

**Mukka Srivatsav**  
B.Tech student, Hyderabad, India  
Email: srivatsavmukka@gmail.com  
GitHub: https://github.com/M-Srivatsav999 

---

## 🪪 License

This project is licensed under the **MIT License**.

---

## ⭐ Future Enhancements

- Multi-language text verification
- Docker deployment
- Mobile-friendly frontend

