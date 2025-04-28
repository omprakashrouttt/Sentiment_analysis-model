from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pdfplumber
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allows frontend requests

# Load the trained model and tokenizer from the extracted folder
MODEL_PATH = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

    sentiment_label = "positive" if prediction.item() == 1 else "negative"
    return sentiment_label, confidence.item()

# API Endpoint for Sentiment Prediction (Text Input)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    print("Received Data:", data)  # Debugging

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    sentiment, confidence = predict_sentiment(text)
    
    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

# API Endpoint for File Upload (for image/PDF text extraction)
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process the file based on its type (PDF or image)
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        sentiment, confidence = predict_sentiment(text)
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from Image using Tesseract OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
