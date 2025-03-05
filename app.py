from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from joblib import load
from model import llm_model, llm_tokenizer, max_length
from predict import Entropy

# Initialize Flask app
app = Flask(__name__)

# Load the trained One-Class SVM model and normalization parameters
classifier = load('oneClassSVM.joblib')
z_data = np.load('zscore.npz')
z_mean, z_std = z_data['z_mean'], z_data['z_std']


def preprocess_text(text):
    """
    Preprocess the input text:
    - Tokenize using the LLM tokenizer.
    - Compute entropy-based features.
    - Normalize features using z-score.
    """
    device = next(llm_model.parameters()).device

    # Tokenize the input text
    tokens = llm_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Forward pass through the model to get logits
    with torch.no_grad():
        logits = llm_model(**tokens).logits

    # Compute entropy
    entD, entL = Entropy.compute_entropy(tokens.input_ids, logits, tokens.attention_mask)

    # Compute features
    feats_list = ['Dmed', 'Lmed', 'Dp05', 'Lstd', 'meanchr']
    mean_chr = len(text) / np.sum(np.isfinite(entL), axis=-1)
    Dmed = np.nanmedian(entD, axis=-1)
    Lmed = np.nanmedian(entL, axis=-1)
    Dp05 = np.nanpercentile(entD, 5, axis=-1)
    Lstd = np.nanstd(entL, axis=-1)

    # Combine features into a single array
    features = np.array([mean_chr, Dmed, Lmed, Dp05, Lstd]).T

    # Normalize features using z-score
    normalized_features = (features - z_mean) / z_std

    return normalized_features


@app.route('/')
def home():
    """
    Render the home page with a form for text input.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether the input text is AI-generated or human-generated.
    """
    try:
        # Get the input text from the request
        text = request.form['text']

        if not text.strip():
            return jsonify({'error': 'Input text cannot be empty'}), 400

        # Preprocess the text and extract features
        features = preprocess_text(text)

        # Predict using the One-Class SVM model
        prediction = classifier.predict(features)[0]

        # Interpret the result
        result = "Human-Generated" if prediction == 1 else "AI-Generated"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)