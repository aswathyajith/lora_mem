from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from basic_prompter import *

app = Flask(__name__)

# # Load models and tokenizers
# models = {
#     "Base Model": {
#         "model": load_model("EleutherAI/pythia-1.4b", None),
#         "tokenizer": AutoTokenizer.from_pretrained("gpt2")
#     },
#     "LoRA Model": {
#         "model": AutoModelForCausalLM.from_pretrained("gpt2-medium"),
#         "tokenizer": AutoTokenizer.from_pretrained("gpt2-medium")
#     },
#     "Full Fine-Tuned Model": {
#         "model": AutoModelForCausalLM.from_pretrained("distilgpt2"),
#         "tokenizer": AutoTokenizer.from_pretrained("distilgpt2")
#     }
# }
base_model, tokenizer = load_model("EleutherAI/pythia-1.4b", None)
lora_model, tokenizer = load_model("EleutherAI/pythia-1.4b", "models/lora/pythia-1.4b/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/checkpoint-180")
full_model, tokenizer = load_model("models/full-ft/pythia-1.4b/lr_2e-6/early_stopping/num_train_4096/bsize_128/checkpoint-80", None)

# Load models and tokenizers
models = {
    "Base Model (pythia-1.4b)": {
        "model": base_model,
        "tokenizer": tokenizer
    },
    "LoRA Fine-tuned (lr=2e-4, r=16)": {
        "model": lora_model,
        "tokenizer": tokenizer
    },
    "Full Fine-tuned (lr=2e-6)": {
        "model": full_model,
        "tokenizer": tokenizer
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    top_k = request.json['topK']
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name, model_data in models.items():
        # Tokenize input text
        inputs = model_data["tokenizer"](text, return_tensors="pt").to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model_data["model"](**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        # Convert to tokens and probabilities
        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = model_data["tokenizer"].decode([idx])
            predictions.append({
                'token': token,
                'probability': round(prob * 100, 2)
            })
        
        results[model_name] = predictions
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)