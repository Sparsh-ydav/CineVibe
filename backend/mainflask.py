from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load model and tokenizer at startup
MODEL_PATH = 'imdb_sentiment_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'

model = None
tokenizer = None
maxlen = 500

def load_resources():
    global model, tokenizer
    try:
        model = load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
        
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Load on startup
load_resources()

@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for a given review text
    
    Expected JSON: {"review": "Your review text here"}
    Returns: {"sentiment": "Positive/Negative", "confidence": 0.95, "score": 0.95}
    """
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({'error': 'No review text provided'}), 400
        
        review_text = data['review']
        
        if not review_text.strip():
            return jsonify({'error': 'Review text is empty'}), 400
        
        # Preprocess the text
        sequence = tokenizer.texts_to_sequences([review_text])
        padded = pad_sequences(sequence, maxlen=maxlen)
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)[0][0]
        
        # Determine sentiment
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'score': float(prediction),
            'review': review_text[:100] + '...' if len(review_text) > 100 else review_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict sentiment for multiple reviews
    
    Expected JSON: {"reviews": ["review1", "review2", ...]}
    """
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({'error': 'No reviews provided'}), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            return jsonify({'error': 'Reviews must be a list'}), 400
        
        # Preprocess all texts
        sequences = tokenizer.texts_to_sequences(reviews)
        padded = pad_sequences(sequences, maxlen=maxlen)
        
        # Make predictions
        predictions = model.predict(padded, verbose=0)
        
        results = []
        for i, pred in enumerate(predictions):
            score = float(pred[0])
            sentiment = "Positive" if score > 0.5 else "Negative"
            confidence = score if score > 0.5 else 1 - score
            
            results.append({
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'score': score,
                'review': reviews[i][:50] + '...' if len(reviews[i]) > 50 else reviews[i]
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if model is None or tokenizer is None:
        print("Warning: Model or tokenizer not loaded. Please check file paths.")
    
    # Run on port 5000
    app.run(debug=True, host='0.0.0.0', port=5001)