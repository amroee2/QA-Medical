import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load tokenizer and model
model_name_or_path = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"
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

# Load the JSON file
json_file_path = "output.json"  # Replace with your JSON file path

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize lists to store results
true_answers = []
predicted_answers = []

# Iterate over each item in the JSON file
for item in tqdm(data):
    question = item["question"]
    context = item["context"]
    true_answer = str(item["answer"])  # Ensure true answer is a string
    
    # Get the predicted answer
    predicted_answer = contextual_qa(question, context)
    
    # Print results for debugging (optional)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"True Answer: {true_answer}")
    print(f"Predicted Answer: {predicted_answer}")
    print("-" * 50)

    # Append to lists
    true_answers.append(true_answer.strip())
    predicted_answers.append(predicted_answer.strip())

# Calculate metrics
precision = precision_score(true_answers, predicted_answers, average='weighted', zero_division=0)
recall = recall_score(true_answers, predicted_answers, average='weighted', zero_division=0)
f1 = f1_score(true_answers, predicted_answers, average='weighted', zero_division=0)
accuracy = accuracy_score(true_answers, predicted_answers)

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")