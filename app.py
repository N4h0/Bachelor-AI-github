from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load questions and the model once
questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)
encoded_questions = model.encode(questions)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Missing question in request'}), 400

    user_question = request.json['question']
    encoded_user_question = model.encode([user_question])[0]
    similarity_scores = cosine_similarity([encoded_user_question], encoded_questions)[0]
    most_similar_question_index = np.argmax(similarity_scores)
    
    # Convert NumPy floats to Python floats for JSON serialization
    nested_list = [[a, float(b)] for a, b in zip(questions, similarity_scores)]
    sorted_nested_list = sorted(nested_list, key=lambda x: x[1], reverse=True)

    most_similar_question = questions[most_similar_question_index]
    print("Returning output to user: ", most_similar_question)
    print("UserQuestion: ", user_question, 'scoreBoard', sorted_nested_list)
    return jsonify(most_similar_question)

if __name__ == '__main__':
    app.run(debug=True)