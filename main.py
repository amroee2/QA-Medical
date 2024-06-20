from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Define the model and tokenizer paths or names
model_name_or_path = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

# Function to perform contextual question answering
def contextual_qa(question, context):
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get the answer from model
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    answer = tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)
    return answer

# Route to handle POST requests for question answering
@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question', '')
    context = data.get('context', '')

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        answer = contextual_qa(question, context)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
